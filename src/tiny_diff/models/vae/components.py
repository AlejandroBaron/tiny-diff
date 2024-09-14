from abc import abstractmethod
from typing import Optional, Union

from torch import nn
from typing_extensions import override

from tiny_diff.models.conv_model import ConvModel
from tiny_diff.models.vae.layers import LatentAlignLayer
from tiny_diff.modules import (
    ConvBlock,
    DownsampleResidualLayer,
    PreNormActConvBlock,
    UpsampleConvBlock,
    UpsampleResidualLayer,
)


class ConvVAEComponent(ConvModel):
    """Class representing a VAE model component (encoder/decoder).

    Args:
        z_channels: channel dimmension size of the latent space.
        attn_channels: at what channels should attention be used.
    """

    @property
    @abstractmethod
    def residual_cls(self) -> Union[DownsampleResidualLayer, UpsampleResidualLayer]:
        """Class of the residual layers."""

    def __init__(
        self,
        z_channels: Optional[int] = None,
        attn_channels: Optional[Union[int, list[int]]] = None,
        input_resolution: int = 96,
        **kwargs,
    ):
        self.z_channels = z_channels
        self.input_resolution = input_resolution
        self.attn_channels = (
            [attn_channels] if isinstance(attn_channels, int) else attn_channels or []
        )
        super().__init__(**kwargs)
        self.conv_layers = nn.ModuleList()
        self._set_conv_layers()

    @property
    def embedding_size(self):
        """Embedding dimension size."""
        return self.input_resolution // (self.factor ** (self.n_conv_layers - 1))

    @property
    def output_channels(self):
        """Output's channel dimension size."""
        return self.z_channels or self.channels[-1]

    @property
    @abstractmethod
    def align_in_ch(self):
        """Channels dimension size of the align layer's input."""

    @property
    @abstractmethod
    def align_out_ch(self):
        """Channels dimension size of the align layer's output."""

    @abstractmethod
    def _set_conv_layers(): ...

    @abstractmethod
    def _in_shape_at_ch(self, ch: int): ...

    def _align_layer(self, **kwargs):
        align_kwargs = self.conv_kwargs(
            in_ch=self.align_in_ch,
            out_ch=self.align_out_ch,
            kernel_size=1,
            stride=1,
            padding="same",
            num_groups=min(self.num_groups, self.z_channels)
            if self.num_groups
            else None,
            scale_factor=None,
            **kwargs,
        )
        layer = LatentAlignLayer(**align_kwargs)
        return layer

    def _attention_at_ch(self, in_ch: int, out_ch: int):
        if in_ch <= out_ch:
            return out_ch in self.attn_channels
        return in_ch in self.attn_channels

    def _append_residuals(self):
        for in_ch, out_ch in zip(self.conv_channels[:-1], self.conv_channels[1:]):
            layer = self._residual_layer(
                in_ch,
                out_ch,
            )
            self.conv_layers.append(layer)

    def _append_align(self):
        if self.z_channels:
            align_layer = self._align_layer()
            self.conv_layers.append(align_layer)

    def _out_shape_at_ch(self, ch: int):
        self._in_shape_at_ch(ch) // self.factor

    def _residual_layer(self, *args, **kwargs):
        kwargs = self.res_kwargs(*args, **kwargs)
        return self.residual_cls(**kwargs)

    @override
    def conv_kwargs(
        self,
        in_ch: int,
        out_ch: int,
        attention: bool = False,
        pop: list[str] = None,
        **kwargs,
    ):
        do_attention = attention or self._attention_at_ch(in_ch, out_ch)
        pop = pop or []
        if do_attention:
            kwargs["attention_factory"] = self.attention_factory
        if (
            not do_attention
            and "attention_factory" not in pop
            and "attention_factory" in kwargs
        ):
            pop = [*pop, "attention_factory"]
        kwgs = super().conv_kwargs(
            attention=attention,
            in_ch=in_ch,
            out_ch=out_ch,
            pop=pop,
            **kwargs,
        )
        return kwgs

    def conv_forward(self, x):
        """Convolutional layer's forward."""
        for layer in self.conv_layers:
            x = layer(x)
        return x

    @override
    def forward(self, x):
        h = self.conv_forward(x)
        return h


class ConvEncoder(ConvVAEComponent):
    """ConvEncoder for a VAE model."""

    residual_cls = DownsampleResidualLayer
    conv_cls = ConvBlock

    @override
    @property
    def align_in_ch(self):
        return self.channels[-1]

    @override
    @property
    def align_out_ch(self):
        return self.z_channels

    def _factor_at_layer(self, layer_number):
        return self.factor**layer_number

    def _in_shape_at_ch(self, in_ch: int, out_ch: int):
        ch = in_ch
        if ch == self.z_channels:
            ch = self.channels[-1]
        if ch == self.input_channels:
            ch = self.conv_channels[0]
        layer_number = self.conv_channels.index(ch)
        return self.input_resolution // self._factor_at_layer(layer_number)

    def _set_conv_layers(self):
        initial_attention = self.conv_channels[0] in self.attn_channels
        self.conv_layers.append(
            ConvBlock(
                **self.conv_kwargs(
                    in_ch=self.input_channels,
                    out_ch=self.conv_channels[0],
                    nonlinearity=None,
                    num_groups=None,
                    scale_factor=None,
                    attention=initial_attention,
                ),
            )
        )
        self._append_residuals()
        self._append_align()


class ConvDecoder(ConvVAEComponent):
    """ConvDecoder for a VAE model."""

    residual_cls = UpsampleResidualLayer
    conv_cls = UpsampleConvBlock

    def __init__(self, final_act: Optional[nn.Module] = None, **kwargs):
        super().__init__(**kwargs)
        self.final_act = final_act or nn.Sigmoid()

    @override
    @property
    def channels(self) -> int:
        return [*self.conv_channels, self.input_channels]

    @override
    @property
    def conv_channels(self) -> int:
        return super().conv_channels[::-1]

    @override
    @property
    def align_in_ch(self) -> int:
        return self.z_channels

    @override
    @property
    def align_out_ch(self) -> int:
        return self.channels[0]

    @override
    @property
    def output_channels(self) -> int:
        return self.channels[-1]

    def _in_shape_at_ch(self, in_ch: int, out_ch: int):
        # TODO: this could be unified for encoder decoder with minmax
        ch = out_ch
        if ch == self.z_channels:
            ch = self.conv_channels[0]
        if ch == self.input_channels:
            ch = self.conv_channels[-1]

        layer_number = self.conv_channels.index(ch)
        return self.input_resolution // self._factor_at_layer(layer_number)

    def _factor_at_layer(self, layer_number: int):
        n = len(self.channels) - layer_number - 1
        return self.factor**n

    def _align_layer(self, **kwargs):
        attention = self.align_in_ch in self.attn_channels
        return super()._align_layer(attention=attention, **kwargs)

    def _set_conv_layers(self):
        self._append_align()
        self._append_residuals()
        final_attention = self.conv_channels[-1] in self.attn_channels
        self.conv_layers.append(
            PreNormActConvBlock(
                **self.conv_kwargs(
                    in_ch=self.conv_channels[-1],
                    out_ch=self.input_channels,
                    nonlinearity=None,
                    scale_factor=None,
                    attention=final_attention,
                )
            )
        )

    @override
    def forward(self, x):
        h = super().forward(x)
        return self.final_act(h)

    @property
    def last_layer(self) -> nn.Module:
        """ConvDecoder's last conv layer."""
        last_block = self.conv_layers[-1]
        return last_block.conv_layer[0]
