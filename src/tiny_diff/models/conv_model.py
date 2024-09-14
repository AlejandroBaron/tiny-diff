from abc import ABC
from typing import Optional

from torch import nn, save

from tiny_diff.modules.layer_factory import LayerFactory


class ConvModel(nn.Module, ABC):
    """Abstract class for any model that uses convolutional layers.

    Provides common init args and properties useful to avoid code duplication.

    Args:
        base_channels: Number of base channels for the model.
        channel_mult: List of multipliers for the base channels in each layer.
        n_res_blocks: Number of residual blocks to use in the model.
        input_channels: Number of input channels for the model (e.g., 3 for RGB images).
        kernel_size: Size of the convolutional kernel.
        factor: Upscaling or downscaling factor used in interpolation layers.
        dropout: Dropout rate to use in the model.
        num_groups: Number of groups for group normalization.
        nonlinearity_factory: Callable that returns a nonlinearity module.
        nonlinearity: Default nonlinearity to use in the model.
        interpolation_mode: Mode used for interpolation (e.g., 'bilinear').
    """

    def __init__(  # noqa: PLR0913
        self,
        base_channels: int = 32,
        channel_mult: Optional[list[int]] = None,
        n_res_blocks: int = 3,
        input_channels: int = 3,
        kernel_size: int = 4,
        factor: float = 2,
        dropout: float = 0.1,
        num_groups: int = 8,
        nonlinearity_factory: Optional[LayerFactory] = None,
        interpolation_mode: str = "bilinear",
        attention_factory: Optional[LayerFactory] = None,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.channel_mult = channel_mult or [1, 2]
        self.n_res_blocks = n_res_blocks
        self.kernel_size = kernel_size
        self.factor = factor
        self.input_channels = input_channels
        self.dropout = dropout
        self.num_groups = num_groups
        self.interpolation_mode = interpolation_mode
        self.nonlinearity_factory = nonlinearity_factory or LayerFactory(cls=nn.SiLU)
        self.attention_factory = attention_factory

    def conv_kwargs(
        self, in_ch: int, out_ch: int, pop: list[str] = None, **kwargs
    ) -> dict:
        """Kwargs to pass to convolutional blocks init."""
        pop = pop or []
        default_kwargs = {
            "in_channels": in_ch,
            "out_channels": out_ch,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": "same",
            "num_groups": self.num_groups,
            "drop_p": self.dropout,
            "scale_factor": self.scale_factor,
            "interpolation_mode": self.interpolation_mode,
            "nonlinearity": self.nonlinearity_factory.layer(),
        }
        kwargs = {**default_kwargs, **kwargs}
        for kwarg in pop:
            kwargs.pop(kwarg)
        return kwargs

    def res_kwargs(self, *args, **kwargs):
        """Kwargs to pass to residual layer's init."""
        if self.attention_factory:
            kwargs["attention_factory"] = self.attention_factory
        return self.conv_kwargs(
            *args,
            n_blocks=self.n_res_blocks,
            **kwargs,
        )

    @property
    def stride(self) -> float:
        """Stride used in convolutional blocks."""
        return 1 if self.interpolation_mode else self.factor

    @property
    def scale_factor(self) -> float:
        """Scale factor used in in interpolation layers."""
        if self.interpolation_mode:
            return self.factor
        return None

    @property
    def conv_channels(self) -> list[int]:
        """Channels in convolutional layers."""
        return [self.base_channels * cmult for cmult in self.channel_mult]

    @property
    def channels(self) -> list[int]:
        """Input channels + Conv channels."""
        return [self.input_channels, *self.conv_channels]

    @property
    def n_conv_layers(self) -> int:
        """How many conv layers does the model have."""
        return len(set(self.channels)) - 1

    def save(self, path):
        """Saves himself at path."""
        return save(self.state_dict(), str(path))
