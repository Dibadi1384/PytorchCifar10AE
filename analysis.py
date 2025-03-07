import torch
import torchvision
import matplotlib.pyplot as plt
import plotly.express as px
from torchsummary import summary

from autoencoder import Autoencoder  
from train import DATASET_PATH, test_set 

def plot_reconstruction_errors_plotly(model_dict):
    latent_dims = sorted(model_dict.keys())
    val_scores = []
    for ld in latent_dims:
        # Assume the first dictionary in the 'val' list contains our "test_loss"
        val_loss = model_dict[ld]["result"]["val"][0]["test_loss"]
        val_scores.append(val_loss)

    fig = px.line(
        x=latent_dims,
        y=val_scores,
        markers=True,
        title="Reconstruction Error Over Latent Dimensionality"
    )
    fig.update_xaxes(type='log', title_text='Latent Dimensionality', tickvals=latent_dims)
    fig.update_yaxes(range=[0, 100], title_text='Reconstruction Error')
    plt.savefig(f"Model_losses_from_{ModelDim}Dim")

def reconstruct(model, dataset, fixed_indices):
    model.eval()
    device = next(model.parameters()).device

    images = [dataset[i][0] for i in fixed_indices]
    images = torch.stack(images, dim=0).to(device)

    with torch.no_grad():
        reconstructions = model(images)

    images = images.cpu()
    reconstructions = reconstructions.cpu()

    combined = torch.cat([images, reconstructions], dim=0)

    grid = torchvision.utils.make_grid(
        combined, 
        nrow=len(fixed_indices), 
        normalize=True, 
        value_range=(-1, 1)
    )

    plt.figure(figsize=(20, 5))
    plt.title(f"Reconstructed Images from {ModelDim}Dim")
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f"Reconstructed_Images_from_{ModelDim}Dim")

if __name__ == "__main__":
    # Load a trained checkpoint
    ModelDim=64
    checkpoint_path = f"saved_models/cifar10_{ModelDim}.ckpt"
    model = Autoencoder.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Summarize model
    summary(model, (3, 32, 32))

    # Image indices to reconstruct
    test_img_indices = [3, 7, 12, 15, 20, 22, 27, 30, 35, 40, 
                 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

    # Reconstruct images from the selected indices
    reconstruct(model, test_set, test_img_indices)

    # model_dict = {
    #     16: {"result": {"val": [{"test_loss": 35.5}]}},
    #     32: {"result": {"val": [{"test_loss": 28.7}]}},
    #     64: {"result": {"val": [{"test_loss": 22.1}]}},
    # }
    # plot_reconstruction_errors_plotly(model_dict)
