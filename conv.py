import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import CTorchClasses as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GaborNet(nn.Module):
    def __init__(self, G_f: torch.Tensor, G_A_dash: torch.Tensor) -> torch.Tensor:
        super(GaborNet, self).__init__()

        self.G_f = G_f
        self.G_A_dash = G_A_dash
        self.seq = nn.Sequential(
                                 nn.Conv2d(in_channels=225, out_channels=45, kernel_size=5, bias=True, dtype=torch.complex64),
                                 C.ComplexRelu(),
                                 C.ComplexDropput(p=0.25),
                                 nn.Conv2d(in_channels=45, out_channels=15, kernel_size=3, bias=True, dtype=torch.complex64),
                                 C.ComplexRelu(),
                                 C.ComplexDropput(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(in_features=15*122*122, out_features=128, bias=True, dtype=torch.complex64),
                                 ).to(DEVICE)


    def gabor_conv_operation(self, img:torch.Tensor, kernel1:torch.Tensor, kernel2:torch.Tensor) -> torch.Tensor:
        split1 = F.conv2d(img, kernel1, padding=(kernel1.shape[-1]//2, kernel1.shape[-2]//2))
        split2 = F.conv2d(img, kernel2, padding=(kernel2.shape[-1]//2, kernel2.shape[-2]//2))

        return torch.concat((split1, split2), dim=1)

    def apply_k1k2filters(self, img:torch.Tensor) -> torch.Tensor:
        img = img.type(torch.complex64)
        G_f_kernels = (self.G_f).unsqueeze(1).to(img.device)
        G_A_dash_kernels = (self.G_A_dash).unsqueeze(1).to(img.device)
        
        k1_out = self.gabor_conv_operation(img, G_f_kernels, G_A_dash_kernels)

        ###########################################################################################################################################
        # NOT OPTIMIZED
        # channel_list = torch.split(k1_out, 1, dim=1)
        # outputs = [self.gabor_conv_operation(channel, G_f_kernels, G_A_dash_kernels) for channel in channel_list]
        # k1k2_out = torch.cat(outputs, dim=1)
        # k1k2_out = torch.cat([self.gabor_conv_operation(k1_out[:, i:i+1, :, :], G_f_kernels, G_A_dash_kernels) for i in range(k1_out.shape[1])], dim=1)
        ###########################################################################################################################################

        # DO CONVOLUTION FOR EVERY CHANNEL BY RESHAPING k1_out AS B*C,1,H,W
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        k1k2_out = self.gabor_conv_operation(k1_out.view(-1,1,img.shape[-2],img.shape[-1]), G_f_kernels, G_A_dash_kernels).view(img.shape[0],-1,img.shape[-2],img.shape[-1])
        
        return k1k2_out


    def forward(self, img:torch.Tensor) -> torch.Tensor:

        # img => (B,C,H,W)
        k1k2_img = self.apply_k1k2filters(img)
        
        out = self.seq(k1k2_img)

        return out # B, C, H, W
