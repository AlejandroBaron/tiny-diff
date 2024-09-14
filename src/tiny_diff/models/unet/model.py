import torch
from torch import nn
from typing_extensions import Optional, override

from tiny_diff.models.conv_model import ConvModel
from tiny_diff.models.unet.layers import (
    DownUNetLayer,
    TimeEmbedding,
    UNetLayer,
    UNetSkipLayer,
    UpUNetLayer,
)
from tiny_diff.models.unet.wrappers import make_fwd_ignore_context
from tiny_diff.modules import (
    ERARBlock,
    LayerFactory,
    PreNormActConvBlock,
    SelfVisualAttention,
)


class UNet(ConvModel):
    """UNet model used for diffusion tasks.

    Args:
        time_channels: channels of the time embedding dimension
        attn_at_layer: list of layer indexes in which attention should be used
        zero_init: whether to init weights to zero or not
    """

    def __init__(
        self,
        time_channels: int = None,
        attn_at_layer: list[bool] = None,
        zero_init: bool = True,
        mid_attention_factory: Optional[LayerFactory] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_at_layer = attn_at_layer or []
        if self.attn_at_layer and len(self.attn_at_layer) != len(self.channel_mult):
            raise ValueError("Length of self.attn_at_layer must match number of layers")
        self.mid_attention_factory = (
            mid_attention_factory
            or self.attention_factory
            or LayerFactory(cls=SelfVisualAttention)
        )

        self.time_channels = time_channels or self.base_channels * 4
        self.zero_init = zero_init
        self.conv_in = PreNormActConvBlock(
            **self.conv_kwargs(
                in_ch=self.input_channels,
                out_ch=self.base_channels,
                nonlinearity=None,
                scale_factor=None,
                drop_p=0,
                num_groups=None,
            )
        )
        # Down blocks
        self.down = nn.ModuleList(self._get_down_blocks())

        # Mid block
        mid_channels = self.channels[-1]
        self.middle = ERARBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            embed_channels=self.time_channels,
            kernel_size=self.kernel_size,
            padding="same",
            num_groups=self.num_groups,
            scale_factor=None,
            attention_factory=self.mid_attention_factory,
        )
        self.up = nn.ModuleList(self._get_up_blocks())

        self.conv_out = PreNormActConvBlock(
            **self.conv_kwargs(
                self.up_channels[-1],
                self.input_channels,
                scale_factor=None,
                attention=None,
                drop_p=0,
            )
        )

        self.time_emb = TimeEmbedding(self.time_channels)

    @property
    def r_channels(self):
        """Channels property reversed."""
        return self.channels[::-1]

    @property
    def down_channels(self) -> list[int]:
        """Channels of the first half of the UNet."""
        return [self.base_channels, *self.conv_channels]

    @property
    def up_channels(self) -> list[int]:
        """Channels of the second half of the UNet."""
        return self.down_channels[::-1]

    def _instantiate_layer(self, cls: type[nn.Module], kwargs) -> nn.Module:
        layer = cls(**kwargs)
        if not layer.attention:
            make_fwd_ignore_context(layer)
        return layer

    def _get_down_blocks(self) -> list[nn.Module]:
        down_in_chs = self.down_channels[:-1]
        down_out_chs = self.down_channels[1:]
        down = []
        for i, (ch_in, ch_out, attn) in enumerate(
            zip(down_in_chs, down_out_chs, self.attn_at_layer)
        ):
            final_block = i == (len(down_in_chs) - 1)
            layer_cls = UNetLayer if final_block else DownUNetLayer
            layer = self._instantiate_layer(
                cls=layer_cls,
                kwargs=self.res_kwargs(
                    ch_in,
                    ch_out,
                    attention=attn,
                    pop=["scale_factor"] if final_block else [],
                ),
            )
            down.append(layer)
        return down

    def _get_up_blocks(self) -> list[nn.Module]:
        up_in_chs = self.up_channels[1:]
        up_out_chs = self.up_channels[:-1]
        attn_at_up_layer = self.attn_at_layer[::-1]
        skip_chs = [up_out_chs[0]] + up_out_chs[:-1]
        # Up blocks
        up = []
        for i, (ch_in, ch_out, attn, ch_skip) in enumerate(
            zip(up_in_chs, up_out_chs, attn_at_up_layer, skip_chs)
        ):
            final_block = i == (len(up_in_chs) - 1)
            layer_cls = UNetSkipLayer if final_block else UpUNetLayer
            layer = self._instantiate_layer(
                cls=layer_cls,
                kwargs=self.up_res_kwargs(
                    ch_in,
                    ch_out,
                    skip_in_channels=ch_skip,
                    skip_out_channels=0,
                    attention=attn,
                    pop=["scale_factor"] if final_block else [],
                ),
            )
            up.append(layer)
        return up

    @override
    def conv_kwargs(
        self, in_ch: int, out_ch: int, pop: list[str] = None, **kwargs
    ) -> dict:
        return super().conv_kwargs(
            in_ch, out_ch, pop, zero_init=self.zero_init, **kwargs
        )

    @override
    def res_kwargs(
        self,
        *args,
        attention: bool = False,
        pop: Optional[list["str"]] = None,
        **kwargs,
    ):
        pop = pop or []
        if not attention and "attention_factory" not in pop and self.attention_factory:
            pop.append("attention_factory")
        return super().res_kwargs(
            *args, embed_channels=self.time_channels, pop=pop, **kwargs
        )

    def up_res_kwargs(self, *args, **kwargs):
        """Kwargs for residual layers of the second half of the net."""
        kwgs = self.res_kwargs(*args, **kwargs)
        kwgs["n_blocks"] += 1
        return kwgs

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        t = self.time_emb(t)

        h = self.conv_in(x)
        h_stack = [h]

        for layer in self.down:
            h, hs = layer(h, t, **kwargs)
            h_stack.extend(hs)

        h = self.middle(h, t, **kwargs)

        for layer in self.up:
            n_pop = len(layer.blocks)
            s, h_stack = h_stack[-n_pop:], h_stack[:-n_pop]
            h = layer(h, s, t, **kwargs)

        return self.conv_out(h)
