from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from typing_extensions import override

from tiny_diff.modules.conv import PreNormActConvBlock
from tiny_diff.modules.layer_factory import LayerFactory
from tiny_diff.utils import match_shape


class RBlock(nn.Module):
    """Block for residual layers.

    See Also:
        https://arxiv.org/pdf/2302.06112.pdf.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_p: float = None,
        zero_init: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = PreNormActConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_factory=None,
            drop_p=None,
            zero_init=False,
            **kwargs,
        )
        self.conv2 = PreNormActConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            attention_factory=None,
            drop_p=drop_p,
            zero_init=zero_init,
            **kwargs,
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )

    @override
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return h


class ERBlock(RBlock):
    """Block for residual layer with embeddings."""

    def __init__(
        self,
        embed_channels: int,
        out_channels: int,
        nonlinearity: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, nonlinearity=nonlinearity, **kwargs)

        proy_layers = [nn.Linear(embed_channels, out_channels)]
        if nonlinearity:
            proy_layers = [deepcopy(nonlinearity), *proy_layers]
        self.proy_emb = nn.Sequential(*proy_layers)

    @override
    def forward(self, x: torch.Tensor, e: torch.Tensor):
        h = self.conv1(x)

        e = self.proy_emb(e)
        e = match_shape(e, like=h)

        h = h + e
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return h


class RABlockABC(nn.Module):
    """Residual block with self attention at the end."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_factory: LayerFactory,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_factory = attention_factory
        self.r1 = self._get_r1_block(**kwargs)
        self.a = self._get_attention()

    @abstractmethod
    def _get_r1_block(self, **kwargs) -> RBlock: ...

    def _get_attention(self, **kwargs):
        attention_head_dim = self.attention_factory.kwargs.get("head_dim", 64)

        num_heads = self.out_channels // attention_head_dim
        num_heads = self.attention_factory.kwargs.get("num_heads", num_heads)

        return self.attention_factory.layer(
            channels=self.out_channels,
            num_heads=num_heads,
            head_dim=attention_head_dim,
            **kwargs,
        )


class RABlock(RABlockABC):
    """Residual block with self attention at the end."""

    @override
    def _get_r1_block(self, **kwargs) -> RBlock:
        return RBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **kwargs,
        )

    @override
    def forward(self, x, **kwargs):
        h = self.r1(x)
        h = self.a(h, **kwargs)
        return h


class ERABlock(RABlockABC):
    """Residual block with self attention at the end."""

    @override
    def _get_r1_block(self, **kwargs) -> RBlock:
        return ERBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **kwargs,
        )

    @override
    def forward(self, x, e, **kwargs):
        h = self.r1(x, e)
        h = self.a(h, **kwargs)
        return h


class ERARBlock(ERABlock):
    """Residual layer with attention in the middle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_factory: LayerFactory,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_factory=attention_factory,
            **kwargs,
        )
        self.r2 = ERBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            **kwargs,
        )

    @override
    def forward(self, x: torch.Tensor, e: torch.Tensor, **kwargs):
        h = super().forward(x, e, **kwargs)
        h = self.r2(h, e)
        return h
