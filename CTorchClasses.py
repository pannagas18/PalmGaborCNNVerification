import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexDropput(nn.Module):
    def __init__(self, p:float, training:bool=True):
        super(ComplexDropput, self).__init__()
        self.dropout = F.dropout
        self.p = p
        self.training = training
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mask = self.dropout(torch.ones_like(x.real), self.p, self.training)
        return x * mask
    
class ComplexRelu(nn.Module):
    def __init__(self):
        super(ComplexRelu, self).__init__()
        self.relu = F.relu
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.relu(x.real) + 1.j * self.relu(x.imag)
        return x
    
class TripletMarginWithComplexDistanceLoss(nn.Module):
    def __init__(self, dist:str, margin:float=1.0, swap:bool=False, reduction:str="mean"):
        super(TripletMarginWithComplexDistanceLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.swap = swap
        self.dist = dist
        if self.dist == "cosine_l2_norm":
            self.distance_function = self.complex_CosineDistance_l2Norm_euclidean
        elif self.dist == "cosine_ang":
            self.distance_function = self.complex_CosineDistance_angular
    
    def complex_CosineDistance_angular(self, x1:torch.Tensor, x2:torch.Tensor, dim:int=1, eps:float=1e-8):
        
        # OUTPUT RANGE [0,1]
        # ANGLE 0DEG = 0
        # ANGLE 90DEG = 0.5
        # ANGLE 180DEG = 1
        
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html#torch.nn.functional.cosine_similarity
        # https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm
        # dot_prd = x1@x2.mT
        # similarity = (dot_prd).squeeze(0)/(max(torch.linalg.vector_norm(x1), eps)*max(torch.linalg.vector_norm(x2), eps))
        # eps = torch.tensor([eps]).to(x1.device)
        # similarity = torch.diagonal((x1@((x2.conj()).mT))/(torch.max(torch.linalg.vector_norm(x1, dim=1), eps)*torch.max(torch.linalg.vector_norm(x2, dim=1), eps)))
        
        # https://math.stackexchange.com/questions/273527/cosine-similarity-between-complex-vectors
        # https://mathoverflow.net/questions/40689/what-is-the-angle-between-two-complex-vectors
        
        eps = torch.tensor([eps]).to(x1.device)
        dot_product = (x1@((x2.conj()).mT))
        magnitude_x1 = torch.max(torch.linalg.vector_norm(x1, dim=1), eps)
        magnitude_x2 = torch.max(torch.linalg.vector_norm(x2, dim=1), eps)
        cosine_similarity = dot_product / (magnitude_x1 * magnitude_x2)
        cosine_similarity = torch.diagonal(cosine_similarity)
        
        # https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance
        # https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
        angular_distance = torch.acos(cosine_similarity) / torch.pi
        
        return angular_distance
        # return (1+1j) - torch.diagonal(cosine_similarity)
    
    def complex_CosineDistance_l2Norm_euclidean(self, x1:torch.Tensor, x2:torch.Tensor):
        
        # OUTPUT RANGE [0,2]
        # ANGLE 0DEG = 0
        # ANGLE 90DEG = 1
        # ANGLE 180DEG = 2
        
        # https://en.wikipedia.org/wiki/Cosine_similarity#L2-normalized_Euclidean_distance
        # https://stats.stackexchange.com/questions/71614/distance-measure-of-angles-between-two-vectors-taking-magnitude-into-account
        # https://en.wikipedia.org/wiki/Cosine_similarity#Properties => cosine distance in terms of euclidean distance
        
        ###################################################
        # taking abs to get the vectors to lie in 1st quadrant of the 2d cartesian plane;
        # no need to divide by 2 for the output as output range => [0,1] (0deg or 180deg, 90deg)
        x1 = x1.abs()
        x2 = x2.abs()
        
        # (???)
        # this might not work as linalg.vector_norm calculates norm for complex numbers using input.abs()
        ###################################################
        
        magnitude_x1 = torch.linalg.vector_norm(x1, dim=1)
        magnitude_x2 = torch.linalg.vector_norm(x2, dim=1)
        norm_x1 = torch.divide(x1, magnitude_x1.unsqueeze(-1))
        norm_x2 = torch.divide(x2, magnitude_x2.unsqueeze(-1))
        dist = (torch.nn.functional.pairwise_distance(norm_x1, norm_x2)**2)/2 # dividing because cosine_dist = (euclidean_dist**2)/2 when vectors are l2 normalized to unit length
        
        return dist # dividing to get it in the range [0,1] fro [0,2] only if not x.abs() # dtype => float
        
    def forward(self, anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor):
        # Check validity of reduction mode
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"{self.reduction} is not a valid value for reduction")

        # Check dimensions
        a_dim = anchor.ndim
        p_dim = positive.ndim
        n_dim = negative.ndim
        if not (a_dim == p_dim and p_dim == n_dim):
            raise RuntimeError(
                f"The anchor, positive, and negative tensors are expected to have "
                f"the same number of dimensions, but got: anchor {a_dim}D, "
                f"positive {p_dim}D, and negative {n_dim}D inputs")

        # Calculate loss
        dist_pos = self.distance_function(anchor, positive)
        dist_neg = self.distance_function(anchor, negative)

        if self.swap:
            dist_swap = self.distance_function(positive, negative)
            dist_neg = torch.minimum(dist_neg, dist_swap)
                
        if self.dist == "cosine_l2_norm":
            loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0) # clamping at 1 is not necessary are vectors are already normalized
        elif self.dist == "cosine_ang":
            loss_real = torch.clamp_min(self.margin + dist_pos.real - dist_neg.real, 0)
            loss_imag = torch.clamp_min(self.margin + dist_pos.imag - dist_neg.imag, 0)
            loss = torch.complex(loss_real, loss_imag) # dtype ==> torch.complex64
            loss = torch.abs(loss) # dtype ==> torch.float32

        # Apply reduction
        if self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "mean":
            return torch.mean(loss)
        else:  # reduction == "none"
            return loss
