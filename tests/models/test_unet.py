from itertools import product

import pytest
from torch import Tensor, randn

from tests.conftest import UNET_BASE_CHANNELS, B, C, H, W
from tiny_diff.models.unet import UNet
from tiny_diff.modules.attention import CrossVisualAttention, SelfVisualAttention
from tiny_diff.modules.layer_factory import LayerFactory


def build_unet(**kwargs) -> UNet:
    """Builds a UNet model."""
    return UNet(
        base_channels=UNET_BASE_CHANNELS,
        n_res_blocks=2,
        num_groups=-2,
        input_channels=C,
        kernel_size=3,
        factor=2,
        dropout=0.1,
        **kwargs,
    )


def get_unet_input(n_layers) -> Tensor:
    """Returns a random input making sure it's downsampleable."""
    k = 2**n_layers
    return randn(B, C, H * k, W * k)


# Define the parameter space
n_layers = [2, 3, 4]
attn_at_layers = [None, -1, -2]
param_combinations = list(product(n_layers, attn_at_layers))


@pytest.mark.parametrize("params", param_combinations)
def test_unet_fwd(params, t_emb: Tensor):
    """Tests that the forward method works under multiple inits."""
    n_layers, attn_at_layer_idx = params

    channel_mult = list(range(1, n_layers + 1))
    attn_at_layers = [False] * n_layers
    if attn_at_layer_idx:
        attn_at_layers[attn_at_layer_idx:] = [True] * (-attn_at_layer_idx)

    # Create the UNet model with the current combination of parameters
    attention_factory = LayerFactory(
        cls=SelfVisualAttention, kwargs={"num_heads": 2, "head_dim": 2}
    )

    unet = build_unet(
        channel_mult=channel_mult,
        attn_at_layer=attn_at_layers,
        attention_factory=attention_factory,
    )
    x = get_unet_input(n_layers=n_layers)

    # Forward pass
    assert unet(x, t_emb).shape == x.shape
    n_attention = sum(attn_at_layers)
    for side in [unet.up, unet.down]:
        assert sum(layer.attention for layer in side) == n_attention


def test_conditional_unet_fwd(attention_context: Tensor, t_emb: Tensor):
    """Tests that the unet works with Causal attention."""
    attention_factory = LayerFactory(
        cls=CrossVisualAttention,
        kwargs={
            "num_heads": 2,
            "head_dim": 2,
            "context_dim": attention_context.shape[-1],
        },
    )

    channel_mult = [1, 2]
    unet = build_unet(
        channel_mult=channel_mult,
        attn_at_layer=[True, False],
        attention_factory=attention_factory,
    )
    x = get_unet_input(n_layers=len(channel_mult))
    assert unet(x, t_emb, context=attention_context).shape == x.shape
