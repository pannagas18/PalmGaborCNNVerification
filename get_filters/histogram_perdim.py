
import torch

def histogram_perdim(input:torch.Tensor, nbins:int) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.detach().cpu()
    hist = torch.empty((input.shape[0],nbins), dtype=input.dtype, device=input.device)
    bin_edges = torch.empty((input.shape[0],nbins+1), dtype=input.dtype, device=input.device)
    bin_centers = torch.empty((input.shape[0],nbins), dtype=input.dtype, device=input.device)

    for d in range(input.shape[0]):
        hist[d,:] = torch.histogram(input[d,:],nbins)[0]
        bin_edges[d,:] = torch.histogram(input[d,:],nbins)[1]
        bin_centers[d,:] = bin_edges[d][:-1] + torch.diff(bin_edges[d])/2
    return hist, bin_centers
