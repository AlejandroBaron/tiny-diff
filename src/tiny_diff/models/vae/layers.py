from typing import Optional

from torch import nn

from tiny_diff.modules.conv import PreNormActConvBlock
from tiny_diff.modules.layer_factory import LayerFactory
from tiny_diff.modules.residual import RBlock


class LatentAlignLayer(nn.Sequential):
    """Hidden layer that proyect into the latent space.

    It's a Residual(Attention)Residual layer, plus
    an align conv block so the input and the output match the expected shapes

    Args:
        in_channels: input's channels
        out_channels: output's channels
        attention: whether to use attention or not
        nonlinearity: nonlinearity to use in the residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_factory: Optional[LayerFactory] = False,
        nonlinearity: Optional[nn.Module] = None,
        **kwargs,
    ):
        r_channels = max(in_channels, out_channels)
        r1 = RBlock(
            in_channels=r_channels,
            out_channels=r_channels,
            nonlinearity=None if attention_factory else nonlinearity,
            **kwargs,
        )

        r2 = RBlock(
            in_channels=r_channels,
            out_channels=r_channels,
            nonlinearity=nonlinearity,
            **kwargs,
        )

        conv_align = PreNormActConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            nonlinearity=None,
            **kwargs,
        )

        attn = []
        if attention_factory:
            attn = [attention_factory.layer(channels=r_channels)]

        blocks = [r1, *attn, r2]
        if in_channels > out_channels:
            blocks = [*blocks, conv_align]
        else:
            blocks = [conv_align, *blocks]

        super().__init__(*blocks)
