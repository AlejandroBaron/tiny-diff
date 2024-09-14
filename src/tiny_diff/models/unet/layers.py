import math

import torch
from torch import nn
from typing_extensions import override

from tiny_diff.constants import CHANNEL_DIM
from tiny_diff.modules import (
    ERABlock,
    ERBlock,
    EResidualLayer,
    Swish,
)
from tiny_diff.modules.residual import ConvResizeResidualLayerABC


class TimeEmbedding(nn.Module):
    """Embedding layer for time in a diffusion process."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.proyection = nn.Sequential(
            nn.Linear(self.channels // 4, self.channels),
            Swish(),
            nn.Linear(self.channels, self.channels),
        )

    @property
    def sin_channels(self) -> int:
        """Number of channels associated with the sin component of the embeddings."""
        return self.channels // 8

    @override
    def forward(self, t: torch.Tensor):
        ch = self.sin_channels
        emb = math.log(10_000) / (ch - 1)
        emb = torch.arange(ch, device=t.device) * -emb
        emb = t.unsqueeze(-1) * emb.unsqueeze(0).exp()
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        return self.proyection(emb)


class UNetLayerMixin:
    """Mixin for UNet's layers without skip connections."""

    @property
    def residual_block_cls(self):
        """Class to use in residual blocks."""
        return ERABlock if self.attention else ERBlock

    def _setup_blocks(self, **kwargs):
        in_ch = self.in_channels
        for _ in range(self.n_blocks):
            self.blocks.append(
                self.residual_block_cls(
                    in_channels=in_ch,
                    out_channels=self.out_channels,
                    scale_factor=None,
                    **self._get_res_kwargs(**kwargs),
                )
            )
            in_ch = self.out_channels

    def _get_res_kwargs(self, **kwargs):
        kwgs = {**self.res_kwargs, **kwargs}
        return kwgs

    def residual_forward(self, h: torch.Tensor, *args, **kwargs):
        """Forward of the residual blocks."""
        hs = []
        for block in self.blocks:
            h = block(h, *args, **kwargs)
            hs.append(h)
        return hs

    def forward(self, x, *args, **kwargs):
        """Forward pass."""
        hs = self.residual_forward(x, *args, **kwargs)
        h = hs[-1]
        return h, hs


class UNetSkipLayerMixin(UNetLayerMixin):
    """Mixin for UNet's layers with skip connections."""

    def _setup_blocks(self, **kwargs):
        for i in range(self.n_blocks):
            in_ch = self.in_channels if (i == self.n_blocks - 1) else self.out_channels
            skip_ch = self.skip_in_channels if i == 0 else self.out_channels
            self.blocks.append(
                self.residual_block_cls(
                    in_channels=in_ch + skip_ch,
                    out_channels=self.out_channels,
                    scale_factor=None,
                    **self._get_res_kwargs(**kwargs),
                )
            )

    @override
    def residual_forward(
        self, h: torch.Tensor, down_hs: list[torch.Tensor], *args, **kwargs
    ) -> list[torch.Tensor]:
        for block in self.blocks:
            dh = down_hs.pop()
            h = torch.cat((h, dh), dim=CHANNEL_DIM)
            h = block(h, *args, **kwargs)
        return h

    @override
    def forward(self, x, *args, **kwargs):
        return self.residual_forward(x, *args, **kwargs)


class UNetLayer(UNetLayerMixin, EResidualLayer):
    """UNet's first half layer with residual layers + embeddings."""


class DownUNetLayer(UNetLayerMixin, ConvResizeResidualLayerABC):
    """UNet layer that performs downsampling."""

    conv_cls = nn.Conv2d

    @override
    def forward(self, x, *args, **kwargs):
        hs = self.residual_forward(x, *args, **kwargs)
        h = self.resize(hs[-1])
        hs.append(h)
        return h, hs


class UNetSkipLayer(UNetSkipLayerMixin, EResidualLayer):
    """UNet's second half layer with residual layers + embeddings + skip connections."""

    def __init__(self, skip_in_channels: int, skip_out_channels, **kwargs):
        self.skip_in_channels = skip_in_channels
        self.skip_out_channels = skip_out_channels
        super().__init__(**kwargs)


class UpUNetLayer(UNetSkipLayerMixin, ConvResizeResidualLayerABC):
    """UNet layer that performs upsampling."""

    conv_cls = nn.ConvTranspose2d

    def __init__(self, skip_in_channels: int, skip_out_channels, **kwargs):
        self.skip_in_channels = skip_in_channels
        self.skip_out_channels = skip_out_channels
        super().__init__(**kwargs)

    @override
    def forward(self, x, down_hs, *args, **kwargs):
        h = self.residual_forward(x, down_hs, *args, **kwargs)
        B, C, H, W = h.shape
        f = int(self.scale_factor)
        h = self.resize(h, output_size=(B, C, H * f, W * f))
        return h
