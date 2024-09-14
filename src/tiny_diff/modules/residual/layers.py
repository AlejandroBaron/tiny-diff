from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn
from typing_extensions import override

from tiny_diff.modules.resampling import Downsample, Interpolation, Upsample
from tiny_diff.modules.residual.blocks import (
    ERABlock,
    ERBlock,
    RABlock,
    RBlock,
)


class ResidualLayerABC(nn.Module, ABC):
    """Residual layer abstract class.

    Args:
        in_channels: input's channel dim size
        out_channels: output's channel dim size
        n_blocks: number of blocks to use.
        attention: whether to include attention or not.
    """

    @property
    @abstractmethod
    def residual_block_cls(self) -> RBlock:
        """Class to use in residual conv blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        self.res_kwargs = kwargs
        self._setup_blocks()

    @property
    def attention(self):
        """Indicates if layer uses attention."""
        return self.res_kwargs.get("attention_factory", None) is not None

    def _setup_blocks(self, **kwargs):
        self.blocks.append(
            self.residual_block_cls(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                scale_factor=None,
                **self.res_kwargs,
                **kwargs,
            )
        )
        for _ in range(1, self.n_blocks):
            self.blocks.append(
                self.residual_block_cls(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    scale_factor=None,
                    **self.res_kwargs,
                    **kwargs,
                )
            )

    def residual_forward(self, h: torch.Tensor, *args, **kwargs):
        """Residual blocks forward."""
        for block in self.blocks:
            h = block(h, *args, **kwargs)
        return h

    @override
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.residual_forward(x, *args, **kwargs)

    @property
    def last_layer(self) -> nn.Module:
        """Last block of the residual layer."""
        return self.blocks[-1]


class ResizeResidualLayerABC(ResidualLayerABC):
    """Residual layer that performs resizing."""

    def __init__(self, scale_factor: float, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)
        self.resize = self._get_resize_layer()

    @abstractmethod
    def _get_resize_layer(self) -> nn.Module: ...

    @override
    def forward(self, *args, **kwargs):
        h = super().forward(*args, **kwargs)
        h = self.resize(h)
        return h


class InterpolationResidualLayerABC(ResizeResidualLayerABC):
    """Residual layer with interpolation resizing."""

    @override
    @property
    def residual_block_cls(self):
        if self.attention:
            return RABlock
        return RBlock

    @property
    @abstractmethod
    def resize_cls(self) -> type[Interpolation]:
        """Class used in resizing layers."""

    def __init__(self, interpolation_mode: str = "bilinear", **kwargs):
        self.interpolation_mode = interpolation_mode
        super().__init__(**kwargs)

    def _get_resize_layer(self) -> nn.Module:
        return self.resize_cls(
            scale_factor=self.scale_factor, mode=self.interpolation_mode
        )


class ConvResizeResidualLayerABC(ResizeResidualLayerABC):
    """Residual layer with convolutional resizing."""

    @property
    @abstractmethod
    def conv_cls(self) -> type[Union[nn.Conv2d, nn.ConvTranspose2d]]:
        """Class to use in convolution layers."""

    def __init__(self, kernel_size: int, **kwargs):
        self.kernel_size = kernel_size
        super().__init__(kernel_size=kernel_size, **kwargs)

    @property
    def padding(self):
        """Padding to use in convolutions."""
        return 1

    def _get_resize_layer(self, **kwargs) -> nn.Module:
        return self.conv_cls(
            self.out_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=int(self.scale_factor),
            padding=self.padding,
            **kwargs,
        )


class ResidualLayer(ResidualLayerABC):
    """Residual layer."""

    @override
    @property
    def residual_block_cls(self):
        if self.attention:
            return RABlock
        return RBlock


class EResidualLayer(ResidualLayerABC):
    """Residual layer with embeddings."""

    @override
    @property
    def residual_block_cls(self):
        if self.attention:
            return ERABlock
        return ERBlock


class DownsampleResidualLayer(InterpolationResidualLayerABC):
    """Residual layer with interpolation downscaling."""

    resize_cls = Downsample


class UpsampleResidualLayer(InterpolationResidualLayerABC):
    """Residual layer with interpolation upscaling."""

    resize_cls = Upsample
