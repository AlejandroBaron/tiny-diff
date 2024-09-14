from typing import Any, Optional

from torch import nn
from typing_extensions import override

from tiny_diff.losses.discriminator.d_loss import DLoss
from tiny_diff.modules import ConvBlock


class Pix2PixDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix.

    Args:
        disc_loss: discriminator loss.
        input_channels: input's channel dimension size.
        base_channels: number indicating the ratio of the convolutional channels.
        n_layers: number of layers.
        ch_mult: multipliers of base_channels.
        conv_kwargs: kwargs for convolutional layers

    See Also:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        disc_loss: Optional[DLoss] = None,
        input_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        ch_mult: int = 1,
        conv_kwargs: dict[str, Any] = None,
    ):
        conv_kwargs = conv_kwargs or {}
        super().__init__()
        self.layers = nn.ModuleList()
        ic = input_channels
        for i in range(n_layers):
            oc = base_channels * (i + 1) * ch_mult
            self.layers.append(
                ConvBlock(in_channels=ic, out_channels=oc, **conv_kwargs)
            )
            ic = oc
        self.disc_loss = disc_loss

    @override
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, x_fake):
        """Computes the discriminator loss."""
        return self.disc_loss(logits_real=self(x), logits_fake=self(x_fake))
