
import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def compute_orientations(image:torch.Tensor, sobel_x:list, sobel_y:list):
    # low pass filter
    filtered_image = torchvision.transforms.functional.gaussian_blur(image, (5,5))

    # SCHARR KERNEL*
    sobel_kernel_x = torch.tensor(sobel_x, dtype=torch.float32).expand(image.shape[1],1,-1,-1).to(DEVICE) # images_tensor.shape[1]
    sobel_kernel_y = torch.tensor(sobel_y, dtype=torch.float32).expand(image.shape[1],1,-1,-1).to(DEVICE) # images_tensor.shape[1]

    # gradient-based operator
    gradient_x = torch.nn.functional.conv2d(filtered_image, sobel_kernel_x, padding=1, groups=image.shape[1])
    gradient_y = torch.nn.functional.conv2d(filtered_image, sobel_kernel_y, padding=1, groups=image.shape[1])

    orientations = torch.arctan2(gradient_y, gradient_x)

    return orientations.view(image.shape[1], 1, image.shape[2],image.shape[3])
