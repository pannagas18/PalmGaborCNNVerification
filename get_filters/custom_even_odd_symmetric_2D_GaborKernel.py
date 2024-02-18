
import torch
import cv2
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PAPER REFERENCE
def custom_even_odd_symmetric_2D_GaborKernel(theta_c: torch.Tensor, mf:int, r:float, a0:float, ktype=cv2.CV_64F):
    
    """theta_c belongs to ThetaC\n
    mf belongs M"""
    
    theta_c = theta_c.detach().cpu().numpy()
    c = np.cos(theta_c)
    s = np.sin(theta_c)

    ksize = (4*(2**mf)+1, 4*(2**mf)+1) # g1, g2
    
    # k in the paper
    # kai  = sqrt(2*log(2)) * (2^param.phai+1) / (2^param.phai-1);
    phai = 1.5 # bandwidth of the gabor
    kai = np.sqrt(2*(np.log(2))) * ((2**phai)+1) / ((2**phai)-1)

    tx = np.arange(1, ksize[0] + 1, dtype=np.float32 if ktype == cv2.CV_32F else np.float64)
    ty = np.arange(1, ksize[1] + 1, dtype=np.float32 if ktype == cv2.CV_32F else np.float64)
    kernel_even, kernel_odd = np.meshgrid(tx, ty)
            
    x_rot = ((a0**-mf)*kernel_even - (4+(2**-mf))/2)*c + ((a0**-mf)*kernel_odd - (4+(2**-mf))/2)*s
    y_rot = -((a0**-mf)*kernel_even - (4+(2**-mf))/2)*s + ((a0**-mf)*kernel_odd - (4+(2**-mf))/2)*c
    
    # even-symmetric part, real
    Ga_even = (((2**-0.5)*(a0**-mf)*np.exp(-((r**2)*(x_rot**2) + y_rot**2)/(2*(r**2)))) * (np.exp(1j*kai*x_rot)-np.exp(-(kai**2)/2))).real

    # odd-symmetric part, imaginary
    Ga_odd = (((2**-0.5)*(a0**-mf)*np.exp(-((r**2)*(x_rot**2) + y_rot**2)/(2*(r**2)))) * (np.exp(1j*kai*x_rot))).imag

    # https://github.com/AngeloUNIMI/PalmNet/blob/d03f2f0df0f72ee5d582e0ed4a40fbaf7b27ec1a/functions_Gabor/gaborArrayFromScales.m#L1
    # refer the above github link, implementation of odd-symmetric part is a bit different from the code (psi is replaced by kai)

    Ga_even = torch.from_numpy(Ga_even).float().to(DEVICE)
    Ga_odd = torch.from_numpy(Ga_odd).float().to(DEVICE)
    return Ga_even, Ga_odd
