import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import glob
torch.set_printoptions(precision=10)

from get_filters.compute_orientations import compute_orientations
from get_filters.custom_even_odd_symmetric_2D_GaborKernel import custom_even_odd_symmetric_2D_GaborKernel
from get_filters.histogram_perdim import histogram_perdim
from get_filters.Sm import getSm

class FixedScale_and_AdaptiveMultiScaleGaborFilters():
    def __init__(self, DEVICE:str, imagedir_path:str, n_images:int, n_steps:int, countW:int, nbins:int, S:int, F:int, M:int, A_dash:int, r:float, a0:float, h1:int, h2:int, sigma:float, lambd:float, gamma=1, psi=0):
        
        self.S = S
        self.imagedir_path = imagedir_path
        self.n_images = n_images
        self.n_steps = n_steps
        images_path = sorted(glob.glob(pathname=f"{self.imagedir_path}/*.bmp"))
        images = np.array([cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in images_path[:self.n_images:self.n_steps]])
        # images = images[idx_mf][None]
        self.DEVICE = DEVICE
        self.images_tensor = transforms.ToTensor()(images).permute(1,2,0).unsqueeze(0).to(self.DEVICE) # C,B,H,W => wtf why?
        
        self.countW = countW
        self.nbins = nbins
        self.F = F
        self.M = M
        self.A_dash = A_dash
        self.r = r
        self.a0 = a0
        self.h1 = h1
        self.h2 = h2
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        
        # SCHARR
        self.sobel_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.sobel_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]

        self.ThetaF = self.get_FixedOrientations()

    def get_images_tensor(self) -> torch.Tensor:
        return self.images_tensor.permute(1,0,2,3) # B,C,H,W
    
    def get_FixedOrientations(self) -> torch.Tensor:
        
        i = torch.arange(1, self.F+1).to(self.DEVICE)
        ThetaF = (i)*(torch.pi/self.F)
        
        return ThetaF

    def get_AdaptiveOrientations(self) -> tuple[torch.Tensor, int]:

        orientations = compute_orientations(self.images_tensor, self.sobel_x, self.sobel_y)

        hist, bin_centers = histogram_perdim(orientations.squeeze(0), self.nbins)

        bin_centers_reshaped = torch.cat(bin_centers.unfold(1,1,1).unbind(), dim=1).to(self.DEVICE)
        hist_reshaped = torch.cat(hist.unfold(1,1,1).unbind(), dim=1).to(self.DEVICE)

        hist_avg = torch.mean(hist_reshaped, axis = 1)
        bin_centers_avg = torch.mean(bin_centers_reshaped, axis = 1)

        indices = torch.flip(torch.argsort(hist_avg), dims=[-1])

        ThetaS = bin_centers_avg[indices][:self.S]
        # ThetaF = self.get_FixedOrientations()
        ThetaC = torch.cat([ThetaS, self.ThetaF]).unique()
        C = len(ThetaC) # cardinality
        
        return ThetaC, C

    def compute_FixedScaleGaborFilters(self) -> np.ndarray:
        
        ThetaF = self.ThetaF.detach().cpu().numpy()
        G_f = np.array([cv2.getGaborKernel(ksize=(self.h1,self.h2), sigma=self.sigma, theta=theta, lambd=self.lambd, gamma=self.gamma, psi=self.psi, ktype=cv2.CV_32F) for theta in ThetaF], dtype=np.complex64)
        return G_f
    
    def compute_MultiScaleGaborFilters(self) -> np.ndarray:

        ThetaC, _ = self.get_AdaptiveOrientations()

        # GABOR BANK WILL ONLY CONTAIN EVEN AND ODD SYMMETRIC GABOR FILTERS
        gabor_bank_even_kernels = {}
        gabor_bank_odd_kernels = {}

        count = 0

        for mf in range(0, self.M):
            for theta_c in ThetaC:
                
                custom_gabor_kernel_even, custom_gabor_kernel_odd = custom_even_odd_symmetric_2D_GaborKernel(theta_c=theta_c, mf=mf, r=self.r, a0=self.a0)
                
                gabor_bank_even_kernels[count] = custom_gabor_kernel_even.expand(1,1,-1,-1)
                gabor_bank_odd_kernels[count] = custom_gabor_kernel_odd.expand(1,1,-1,-1)                
                
                count+=1
        
        # permute self.images_tensor
        _, Sm_indices = getSm(self.images_tensor.permute(1,0,2,3), gabor_bank_even_kernels, gabor_bank_odd_kernels)

        Sm_indices_sliced = torch.slice_copy(Sm_indices, dim=-1, start=0, end=self.countW)

        def get_counter(Sm_indices_sliced: torch.Tensor, ind: torch.Tensor):
            return torch.sum(((ind * 128 * 128) < Sm_indices_sliced.unsqueeze(1)) & (Sm_indices_sliced.unsqueeze(1) < ((ind + 1) * 128 * 128)), dim=2)

        ind = torch.arange(60).unsqueeze(0).to(self.DEVICE)
        num_used_wavelets = get_counter(Sm_indices_sliced.unsqueeze(2), ind).squeeze(1).sum(0)

        # norm = (num_used_wavelets/sum(num_used_wavelets))
        # plt.bar(torch.arange(len(norm)), norm)
        # plt.title(f"len_norm:{len(norm)}, M:{0,self.M-3}")
        
        bestWavelets_indices = torch.sort(num_used_wavelets, descending=True).indices[:self.A_dash].tolist()

        # print([(gabor_bank_even_kernels[key].shape, gabor_bank_odd_kernels[key].shape) for key in bestWavelets_indices])

        G_A_dash = np.array([torch.complex(gabor_bank_even_kernels[key], gabor_bank_odd_kernels[key]).squeeze(0,1).detach().cpu().numpy() for key in bestWavelets_indices])
        # G_A_dash = [torch.complex(gabor_bank_even_kernels[key], gabor_bank_odd_kernels[key]).squeeze(0,1).detach().cpu().numpy() for key in bestWavelets_indices]

        return G_A_dash

# BINS = 100
# tensor([ 1.5392686129,  1.6020957232,  0.0314141139, -0.0314131528,
#          1.3507866859,  1.2879593372,  1.2251321077,  1.4136139154,
#          1.1623048782,  1.4764410257]
# BINS = 45
# tensor([1.5357780457e+00, 4.7828751804e-07, 1.2565457821e+00, 1.3961617947e+00,
#         1.1169296503e+00, 9.7731339931e-01, 1.6753941774e+00, 8.3769726753e-01,
#         6.9808119535e-01, 1.8150104284e+00])

# import timeit
# print(timeit.timeit(AdpativeOrientations(10, "dataset/session1", nbins=100).get_AdaptiveOrientations, number=10))
# print(AdpativeOrientations(10, "dataset/session1", nbins=100).get_AdaptiveOrientations())

# F = 10
# h1, h2 = 35, 35 # 50,50
# sigma = 4.34
# # sigma = 5.6179
# lambd = 1/0.11
# # lambd = 0.11
# S = 10
# M = int(np.log2(128/2))-3
# r = 1
# a0 = 2
# A_dash = 5

# obj = FixedScale_and_AdaptiveMultiScaleGaborFilters(imagedir_path="dataset/session2", n_images=1000, n_steps=10,
#                                                     countW=10000,
#                                                     nbins=100,
#                                                     S=S, F=F, M=M, A_dash=A_dash,
#                                                     r=r, a0=a0,
#                                                     h1=h1, h2=h2, sigma=sigma, lambd=lambd)

# print(obj.compute_FixedScaleGaborFilters().shape)
# print(obj.compute_MultiScaleGaborFilters().shape)
