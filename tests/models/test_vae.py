from itertools import product

import pytest
from torch import nn, randn

from tests.conftest import VAE_BASE_CHANNELS, B, C, H, W
from tiny_diff.losses import BetaPolicy, VAELoss
from tiny_diff.models.vae import VAE, ConvVAE
from tiny_diff.modules.attention import SelfVisualAttention
from tiny_diff.modules.layer_factory import LayerFactory


def build_vae(input_resolution: int = H, **kwargs) -> VAE:
    """Instantiates a VAE model."""
    vae_loss = VAELoss(
        recon_l=nn.L1Loss(), kl_w_policy=BetaPolicy(a=1e-6, b=6e-2, offset=40, slope=2)
    )
    non_linearity_factory = LayerFactory(
        cls=nn.LeakyReLU,
        kwargs={"negative_slope": 0.03},
    )
    component_kwargs = {
        "base_channels": VAE_BASE_CHANNELS,
        "n_res_blocks": 2,
        "num_groups": -1,
        "input_channels": 3,
        "kernel_size": 4,
        "input_resolution": input_resolution,
        "factor": 2,
        "dropout": 0.0,
        "z_channels": 4,
        "interpolation_mode": "bicubic",
        "nonlinearity_factory": non_linearity_factory,
        **kwargs,
    }
    return ConvVAE(vae_loss=vae_loss, component_kwargs=component_kwargs)


# Define the parameter space
n_layers = [2, 3, 4]
attn_at_layers = [None, -1, -2]
param_combinations = list(product(n_layers, attn_at_layers))


@pytest.mark.parametrize("params", param_combinations)
def test_vae_fwd(params):
    """Tests that the VAE forward works under multiple inits."""
    n_layers, attn_at_layer_idx = params

    channel_mult = list(range(1, n_layers + 1))
    attn_channels = []
    if attn_at_layer_idx:
        attn_channels = [VAE_BASE_CHANNELS * channel_mult[attn_at_layer_idx]]
    # Create the UNet model with the current combination of parameters
    attention_factory = LayerFactory(
        cls=SelfVisualAttention, kwargs={"num_heads": 2, "head_dim": 2}
    )

    k = 2**n_layers
    x = randn(B, C, H * k, W * k)
    vae = build_vae(
        channel_mult=channel_mult,
        attn_channels=attn_channels,
        attention_factory=attention_factory,
        input_resolution=H * k,
    )

    # Forward pass
    y = vae(x)
    assert y["x_hat"].shape == x.shape
    assert y["z"].shape == vae.latent_sample(B).shape
    for component in [vae.encoder, vae.decoder]:
        assert sum(
            getattr(layer, "attention", False) for layer in component.conv_layers
        ) == len(attn_channels)
