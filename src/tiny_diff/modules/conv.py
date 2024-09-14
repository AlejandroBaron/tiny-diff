from typing import Optional

from torch import nn
from typing_extensions import override

from tiny_diff.modules.layer_factory import LayerFactory
from tiny_diff.modules.norm import GroupNorm
from tiny_diff.modules.resampling import Downsample, Upsample


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ConvBlock(nn.Sequential):
    """See https://paperswithcode.com/method/fixup-initialization."""

    conv_cls = nn.Conv2d
    interpolation_cls = Downsample

    def __init__(  # noqa: PLR0913
        self,
        scale_factor: float = None,
        stride: int = 1,
        padding: int = 1,
        interpolation_mode: str = "bilinear",
        drop_p: float = None,
        num_groups: int = None,
        attention: bool = False,
        attention_layer: Optional[nn.Module] = None,
        attention_factory: Optional[LayerFactory] = None,
        nonlinearity: nn.Module = None,
        in_channels: int = 8,
        out_channels: int = 8,
        zero_init: bool = False,
        **kwargs,
    ) -> None:
        self.scale_factor = scale_factor
        self.zero_init = zero_init
        self._interpolation_mode = interpolation_mode
        self._stride = stride
        self._padding = padding
        self.attention = attention
        layers = self._layers(
            drop_p=drop_p or 0.0,
            num_groups=num_groups,
            attention_factory=attention_factory,
            attention_layer=attention_layer,
            nonlinearity=nonlinearity,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
        super().__init__(*layers)

    def _layers(
        self,
        drop_p: float = 0.2,
        num_groups: int = None,
        attention_factory: Optional[LayerFactory] = False,
        attention_layer: Optional[nn.Module] = None,
        nonlinearity: nn.Module = None,
        in_channels: int = 8,
        out_channels: int = 8,
        **kwargs,
    ):
        layers = []

        ic = in_channels
        oc = out_channels

        conv_layer = self.conv_cls(
            in_channels=ic,
            out_channels=oc,
            stride=self.stride,
            padding=self.padding,
            **kwargs,
        )
        if self.zero_init:
            conv_layer = zero_module(conv_layer)
        layers.append(conv_layer)

        layers = self._add_norm_act(
            layers=layers,
            ic=ic,
            oc=oc,
            num_groups=num_groups,
            nonlinearity=nonlinearity,
            drop_p=drop_p,
        )

        if self.attention and attention_factory:
            attn_layer = attention_layer or attention_factory.layer(channels=oc)
            layers.append(attn_layer)

        if self.scale_factor:
            layers.append(
                self.interpolation_cls(
                    scale_factor=self.scale_factor, mode=self.interpolation_mode
                )
            )
        return layers

    def _add_norm_act(
        self,
        layers,
        ic: int,
        oc: int,
        num_groups: int,
        nonlinearity: nn.Module,
        drop_p: float,
    ):
        if num_groups:
            layers.append(GroupNorm(num_groups=num_groups, num_channels=oc))
        if nonlinearity:
            layers.append(nonlinearity)
        if drop_p > 0:
            layers.append(nn.Dropout2d(p=drop_p))
        return layers

    @property
    def stride(self):
        """Stride used in conv layer."""
        return 1 if self.scale_factor else self._stride

    @property
    def padding(self):
        """Padding used in conv layer."""
        return "same" if self.scale_factor else self._padding

    @property
    def conv_layer(self):
        """Returns the conv layer."""
        for layer in self:
            if isinstance(layer, self.conv_cls):
                return layer
        return None

    @property
    def interpolation_mode(self):
        """How is interpolation being performed."""
        if self.scale_factor:
            return self._interpolation_mode
        raise ValueError("Interpolation mode not available without scale_factor")


class PreNormActConvBlock(ConvBlock):
    """Conv blcok with BN and Nonlinearity before conv instead of after.

    See Also:
        https://arxiv.org/pdf/1603.05027.pdf.
    """

    def _add_norm_act(
        self,
        layers,
        ic: int,
        oc: int,
        num_groups: int,
        nonlinearity: nn.Module,
        drop_p: float,
    ):
        prepend_layers = []
        if num_groups:
            prepend_layers.append(GroupNorm(num_groups=num_groups, num_channels=ic))
        if nonlinearity:
            prepend_layers.append(nonlinearity)
        if drop_p > 0:
            layers.append(nn.Dropout2d(p=drop_p))
        return prepend_layers + layers


class UpsampleConvBlock(ConvBlock):
    """Convolutional block that upscales the output."""

    interpolation_cls = Upsample

    def __init__(self, scale_factor: float = None, **kwargs) -> None:
        scale_factor = 1 / scale_factor if scale_factor else scale_factor
        super().__init__(scale_factor=scale_factor, **kwargs)

    @property
    def conv_cls(self):
        """Class used in conv layers."""
        return super().conv_cls if self.scale_factor else nn.ConvTranspose2d

    @property
    @override
    def padding(self):
        if self.scale_factor:
            return "same"
        if super().padding == "same":
            return 1
        return super().padding
