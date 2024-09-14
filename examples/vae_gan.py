from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch.nn import LeakyReLU
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_diff.losses import HingeDLoss
from tiny_diff.models.discriminator import Pix2PixDiscriminator
from tiny_diff.models.vae import ConvVAE
from tiny_diff.models.vae.gan import VAEGAN
from tiny_diff.modules.layer_factory import LayerFactory

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Config to use throughout the example."""

    learning_rate: float = 1e-6
    output_path: Path = Path("outputs")
    input_path: Path = Path("inputs")
    lr_warmup_steps: int = 500
    num_epochs: int = 10
    train_batch_size: int = 16
    ckpt_save_pattern: str = "ckpt_{epoch}.pt"
    checkpoints_dir: Path = Path("checkpoints")
    in_channels: int = 3
    sample_size: int = 96
    plot_each_steps: int = 500
    num_train_steps: int = 1000


def loss_backward(loss, optimizer):
    """Backward loss computation."""
    optimizer.zero_grad()
    loss.backward()
    parameters = list(optimizer.param_groups[0]["params"])
    torch.nn.utils.clip_grad_norm_(parameters, 2)
    optimizer.step()


config = Config()

batch_size = config.train_batch_size
epochs = config.num_epochs
img_size = config.image_size
plot_each = config.save_image_epochs
limit_samples = config.limit_samples
precheck_invalid = False
print(device)


nlf = LayerFactory(cls=LeakyReLU, kwargs={"negative_slope": 0.03})
vae = ConvVAE(
    transform=...,  # add your input transform if you need so
    component_kwargs={
        "base_channels": 32,
        "channel_mult": [1, 2, 4],
        "n_res_blocks": 2,
        "num_groups": -1,
        "input_channels": 3,
        "kernel_size": 4,
        "factor": 2,
        "dropout": 0.0,
        "attn_channels": [],
        "z_channels": 4,
        "interpolation_mode": "bicubic",
        "nonlinearity_factory": nlf,
    },
)

# Initialize the nonlinearity
nonlinearity = LeakyReLU(negative_slope=0.05)

# Initialize the discriminator model
discriminator = Pix2PixDiscriminator(
    base_channels=16,
    n_layers=3,
    disc_loss=HingeDLoss(),
    conv_kwargs={
        "kernel_size": 3,
        "nonlinearity": nonlinearity,
        "num_groups": -1,
        "drop_p": 0.0,
        "stride": 2,
    },
)


fs = config.checkpoints_dir
pattern = config.ckpt_save_pattern

model = VAEGAN(
    vae=vae,
    discriminator=discriminator,
    disc_weight=config.disc_weight,
    gen_warmup=config.gen_warmup,
)


vae_optimizer = AdamW(
    list(vae.parameters()),
    lr=config.learning_rate,
)
disc_optimizer = AdamW(list(discriminator.parameters()), lr=3e-4, eps=1e-8)


train_dataset = ...
val_dataset = ...
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(f"# Train records {len(train_dataset)}")

for epoch in range(epochs):
    train_disc = epoch >= model.gen_warmup
    for x in tqdm(train_data_loader, desc=f"Epoch {epoch}", miniters=200):
        x = x.to(device)  # noqa: PLW2901
        # Generator
        vae.train()
        discriminator.eval()
        loss = model.loss(x, epoch, mode="generator")
        mlflow.log_metrics({"train_" + k: v for k, v in loss.items()})
        loss_backward(loss["loss"], vae_optimizer)
        # Discriminator
        if train_disc:
            vae.eval()
            discriminator.train()
            loss = model.loss(x, epoch, mode="discriminator")
            mlflow.log_metric("train_discriminator", loss.item())
            loss_backward(loss, disc_optimizer)

    vae.eval()
    discriminator.eval()
    with torch.no_grad():
        val_recon = 0
        for x in tqdm(val_data_loader):
            x = x.to(device)  # noqa: PLW2901
            loss = model.loss(x, epoch, mode="generator")
            val_recon += loss["recon_loss"].mean().item() / len(val_data_loader)
            mlflow.log_metrics({"val_" + k: v for k, v in loss.items()})
            if train_disc:
                loss = model.loss(x, epoch, mode="discriminator")
                mlflow.log_metric("val_discriminator", loss.item())

        if epoch % plot_each == 0:
            vae.save(
                config.checkpoints_dir / config.ckpt_save_pattern.format(epoch=epoch)
            )

        print("LR", vae_optimizer.param_groups[0]["lr"])
