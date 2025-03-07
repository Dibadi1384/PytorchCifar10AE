import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from autoencoder import Autoencoder  
from train import test_set  #

def compute_fid(model, images, device):
    """
    Given a batch of images (as a tensor in CHW format, normalized to [-1, 1]),
    computes the Frechet Inception Distance (FID) between the real images and the reconstructions.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        recons = model(images).cpu()
    
    def rescale_to_uint8(x):
        x = (x + 1) / 2.0
        x = (x * 255).clamp(0, 255).to(torch.uint8)
        return x

    images_uint8 = rescale_to_uint8(images.cpu())
    recons_uint8  = rescale_to_uint8(recons)
    
    if images_uint8.shape[1] == 1:
        images_uint8 = images_uint8.repeat(1, 3, 1, 1)
    if recons_uint8.shape[1] == 1:
        recons_uint8 = recons_uint8.repeat(1, 3, 1, 1)
    
    # Compute FID
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    fid_metric.reset()
    fid_metric.update(images_uint8, real=True)
    fid_metric.update(recons_uint8, real=False)
    fid_value = fid_metric.compute().item()
    return fid_value, recons_uint8

def visualize_reconstructions(original, reconstructed, title="Reconstructed Images"):
    """
    Displays a grid of original vs. reconstructed images.
    Both inputs should be torch tensors in CHW format and in the range [-1, 1].
    """
    original = ((original.cpu() + 1) / 2.0).clamp(0, 1)
    reconstructed = ((reconstructed.cpu() + 1) / 2.0).clamp(0, 1)
    
    imgs = torch.stack([original, reconstructed], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=original.shape[0], normalize=True, value_range=(0, 1))
    grid = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig(f"FIDS_{ModelDim}")
    
if __name__ == "__main__":
    # Define the checkpoint path 
    ModelDim=64
    checkpoint_path = f"saved_models/cifar10_{ModelDim}.ckpt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model from:", checkpoint_path)
    model = Autoencoder.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # Number of samples
    num_images = 500
    test_images = [test_set[i][0] for i in range(num_images)]
    test_images_tensor = torch.stack(test_images, dim=0)

    # Compute the overall FID between original test images and their reconstructions.
    fid_value, recons_uint8 = compute_fid(model, test_images_tensor, device)
    print("Overall FID score on the sample set:", fid_value)

    # Visualize a few reconstructions.
    vis_batch = test_images_tensor[:8]
    with torch.no_grad():
        vis_recons = model(vis_batch.to(device)).cpu()
    
    visualize_reconstructions(vis_batch, vis_recons, title=f"Reconstructed Images (FID: {fid_value:.2f})")
