# %%

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn import LeakyReLU
from torchtune.modules import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from tiny_diff.models import UNet
from tiny_diff.models.vae import ConvVAE
from tiny_diff.modules.attention import SelfVisualAttention
from tiny_diff.modules.layer_factory import LayerFactory
from tiny_diff.schedulers import LatentDDPMScheduler

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


config = Config()


def get_latent_std(vae, dataloader, n_batches: int = 1):
    """Latent scaling factor according to the original paper."""
    std = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x = next(iter(dataloader))["image"].to(device)
            std += vae.forward_encode(x)["z"].std() / n_batches
    return std


def gen_and_plot_examples(vae, output_name: str, latent_scale_factor: float):
    """Generate and plot new images."""
    unet = model
    generator = torch.Generator(device=device).manual_seed(123)
    unet.eval()
    with torch.no_grad():
        if isinstance(config.sample_size, int):
            output_shape = (
                16,
                config.in_channels,
                config.sample_size,
                config.sample_size,
            )
        output = torch.randn(
            output_shape, generator=generator, device=torch.device(device)
        )

        for t in tqdm(noise_scheduler.timesteps):
            epsilon_hat = unet(output, t.to(device))
            output = noise_scheduler.step(epsilon_hat, t, output, generator=generator)[
                "z"
            ]

        x = vae.forward_decode(output / latent_scale_factor)["x_hat"]
        x = x.permute(0, 2, 3, 1).clip(0, 1).cpu().detach().numpy()
        x = x * 255
        print("Generated image shape:", x.shape)
        x = [Image.fromarray(xi.astype("int")).convert("RGB") for xi in x]
        image_grid = make_grid(x, 4, 4)

        test_dir = config.output_path / "samples"
        test_dir.mkdir(parents=True, exist_ok=True)
        output_file = test_dir / f"{output_name}.png"
        image_grid.save(str(output_file), dpi=(96 * 4, 96 * 4))
        print(output_file)
        plt.imshow(image_grid)
        plt.show()
    unet.train()


def make_grid(images, rows, cols):
    """Pastes images as grid."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


dataset = ...  # Use your torch Dataset


train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)


# Initialize the attention factory
attention_factory = LayerFactory(cls=SelfVisualAttention, kwargs={"num_groups": -1})

# Initialize the UNet model
model = UNet(
    base_channels=128,
    channel_mult=[1, 1, 2, 2, 4, 4],
    attn_at_layer=[False, False, False, False, True, False],
    n_res_blocks=3,
    num_groups=-1,
    input_channels=3,
    kernel_size=3,
    factor=2,
    dropout=0.0,
    attention_factory=attention_factory,
).to(device)


# VAE model
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
).load_state_dict(
    ...  # Use a pretrained vae
)
vae.eval()
latent_scale_factor = 1 / get_latent_std(vae, train_dataloader, 2)


noise_scheduler = LatentDDPMScheduler(beta_start=0.00085, beta_end=0.012)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
scaler = torch.amp.GradScaler()


for epoch in range(config.num_epochs):
    for step, batch in enumerate(train_dataloader):
        model.train()
        clean_images = batch.to(device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        nds = noise_scheduler.num_train_steps
        timesteps = (
            torch.randint(
                0,
                nds,
                (bs,),
                device=clean_images.device,
            )
            .long()
            .to(device)
        )

        # bin timesteps for loss analysis
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with torch.autocast(device_type=device, dtype=torch.float16):
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()
        optimizer.zero_grad()
        if (step % config.plot_each_steps) == 0 and step:
            gen_and_plot_examples(vae, f"{epoch}_{step}")

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "epoch": epoch,
        }
        mlflow.log_metrics(logs)
    gen_and_plot_examples(vae, f"{epoch}_final")
    model.save(config.checkpoints_dir / config.ckpt_save_pattern.format(epoch=epoch))
