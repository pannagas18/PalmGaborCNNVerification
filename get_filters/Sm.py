
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def getSm(images_tensor:torch.Tensor, gabor_bank_even_kernels:dict, gabor_bank_odd_kernels:dict) -> tuple[torch.Tensor, torch.Tensor]:

    I_even = torch.empty(len(gabor_bank_even_kernels), *images_tensor.shape).to(DEVICE)

    for idx, kernel in enumerate(list(gabor_bank_even_kernels.values())):
        I_even[idx] = torch.nn.functional.conv2d(images_tensor, kernel, padding=kernel.shape[-1]//2, groups=images_tensor.shape[1])
    
    I_odd = torch.empty(len(gabor_bank_odd_kernels), *images_tensor.shape).to(DEVICE)

    for idx, kernel in enumerate(list(gabor_bank_odd_kernels.values())):
        I_odd[idx] = torch.nn.functional.conv2d(images_tensor, kernel, padding=kernel.shape[-1]//2, groups=images_tensor.shape[1])

    I_x = (I_even**2+I_odd**2).permute(1,0,2,3,4)

    I_x_reshaped = I_x.flatten(1,-1) # creating Sm for every image (Image, Sm)

    Sm, Sm_indices = torch.sort(I_x_reshaped, dim=-1, descending=True)

    return Sm.to(DEVICE), Sm_indices.to(DEVICE)
    # return torch.sort(I_x_reshaped, dim=-1, descending=True)

