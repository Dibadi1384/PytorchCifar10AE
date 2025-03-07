import os
import torch
import pytorch_lightning as pl
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import our autoencoder classes
from autoencoder import Autoencoder

# -----------------------
# Some configuration
# -----------------------
DATASET_PATH = "data"
CHECKPOINT_PATH = "saved_models"

# Setting the seed for reproducibility
pl.seed_everything(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prepare datasets
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

# Define dataloaders
train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)


def get_train_images(num):
    # Returns a stack of `num` images from the training set
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


class GenerateCallback(pl.Callback):
    """
    Simple callback to generate images at the end of certain epochs.
    """
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot or log reconstructions here if needed
            # Example: using tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_cifar(latent_dim):
    """
    Trains (or loads a pre-trained) autoencoder with the specified `latent_dim`.
    Returns the model and dictionary containing test/val losses.
    """
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}_NLL"),
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=100,  # Decrease epochs for quick tests, set 500 for full training
        callbacks=[
            ModelCheckpoint(save_weights_only=False),
            #GenerateCallback(get_train_images(8), every_n_epochs=2),
            LearningRateMonitor("epoch")
        ]
    )

    # If a checkpoint exists, load the model
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}_NLL.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)

    # Otherwise create new autoencoder
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)

    # Test the model
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result, "test": test_result}
    return model, result


if __name__ == "__main__":
    #change this
    latent_dim=256

    model_ld, result_ld = train_cifar(latent_dim)
        