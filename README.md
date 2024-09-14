[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)

<img src="assets/logo.jpg" alt="Alt text" width="128" height="128">

# Tiny Diff (Beta)



This project attempts to provide a minimal implementation of diffusion models written in pure python, without huggingface dependencies.

It aims to:

- Be object oriented to enable end users to extend the SDK for their custom workflows
- Be, as it name states, easy to use (hence focused in simple usecases)
- Have as little duplicated code as possible.
- Be modular so users can reuse layers, blocks and utilities

You can check more info at the [documentation site](https://tiny-diff.readthedocs.io/en/latest/).
Conditional diffusion is still to be included.

At the moment, all the necessary modules to build the original [latent diffusion model](https://arxiv.org/abs/2112.10752) are included

# Installation

Simply run 
```bash
pip install tiny-diff
```


# Quickstart

Check the `examples` folder for a general guideline on how to train Variational Autoencoders + GAN and using them as embedders for Latent Diffusion models

Esentially, the repo provides three groups of models right now

## UNets

Used for diffusion models, they follow the architecture of Huggingface Diffusers implementation
```python
from tiny_diff.models import UNet

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
)
```

## Variational Autoencoders
Used for latent diffusion models as a way to compress the information into an embedded space.
```python
from tiny_diff.models.vae import ConvVAE
from tiny_diff.losses import VAELoss, BetaPolicy
from torch.nn import L1Loss, LeakyReLU
from tiny_diff.modules.layer_factory import LayerFactory

nonlinearity_factory = LayerFactory(cls=LeakyReLU, kwargs={"negative_slope": 0.03})

vae_loss = VAELoss(recon_l=L1Loss(), kl_w_policy=BetaPolicy(a=1e-6, b=6e-2, offset=40, slope=2))
conv_vae_model = ConvVAE(
    transform=...,  # A callable to apply on the input
    vae_loss=vae_loss,
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
        "nonlinearity_factory": nonlinearity_factory,
    },
)

print(conv_vae_model)
```

## Schedulers

These guide the diffusion process to add and remove noise.

```python
from tiny_diff.schedulers import LatentDDPMScheduler, DDPMScheduler

scheduler = LatentDDPMScheduler(beta_start=0.00085, beta_end=0.012)
```

## Submodules

The framework is modular so you can reuse whatever component you want. Here are some examples:

```python
from tiny_diff.conv import ConvBlock, PreNormActConvBlock, UpsampleConvBlock
from tiny_diff.layer_factory import LayerFactory
from tiny_diff.nonlinearity import Swish
from tiny_diff.resampling import Downsample, Upsample
from tiny_diff.residual import (
    DownsampleResidualLayer,
    ERABlock,
    ERARBlock,
    ERBlock,
    EResidualLayer,
    InterpolationResidualLayerABC,
    RBlock,
    UpsampleResidualLayer,
)
```

### Layer Factory
One simple yet unique element of this framework is the layer factory, which is a simple implementation of the
[Factory pattern](https://refactoring.guru/design-patterns/factory-method). This allows to change the attention
or non linearity layers from a single init param while avoiding aliasing.

```python
from tiny_diff.modules.layer_factory import LayerFactory

nonlinearity_factory = LayerFactory(cls=LeakyReLU, kwargs={"negative_slope": 0.03})
```
If you look at the VAE or UNet model, it's trivial to see how to change the attention or non_linearity throughout the
whole network.

## Citation

If you use this repository in your research, please cite it as follows:

### Citation for the Repository

```bibtex
@misc{YourRepoName,
  author = {Alejandro Baron},
  title = {TinyDiff: A hackable pure-torch diffusion implementation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AlejandroBaron/tiny-diff}},
}
