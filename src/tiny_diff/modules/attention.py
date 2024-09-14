from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from tiny_diff.modules.norm import GroupNorm


class AttnFlatten(nn.Flatten):
    """Flattens a 4D BCHW tensor for visual attention."""

    def __init__(self, start_dim: int = 2, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)

    @override
    def forward(self, x, *args, **kwargs):
        flat = super().forward(x, *args, **kwargs)  # B x C x (H*W)
        return flat.permute(0, 2, 1)  # B x (H*W) x C


class VisualAttention(nn.Module):
    """Multihead attention for visual transformers."""

    def __init__(
        self,
        channels: int,
        kv_channels: Optional[int] = None,
        head_dim: int = 64,
        num_heads: int = 8,
        proy_bias: bool = True,
        num_groups: int = -1,
        dropout: Optional[float] = None,
        rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kv_channels = kv_channels or channels
        self.proy_bias = proy_bias
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.rescale_factor = rescale_factor
        self._post_init()

        self.preproc = nn.ModuleDict()
        self.preproc["q"] = self._get_q_preproc()
        self.preproc["v"] = self._get_k_preproc()
        self.preproc["k"] = self._get_v_preproc()

        self.proj = nn.ModuleDict()
        self.proj["q"] = self._get_q_proy()
        self.proj["k"] = self._get_k_proy(in_features=self.kv_channels)
        self.proj["v"] = self._get_v_proy(in_features=self.kv_channels)

        self.out_proj = self._get_out_proj()

    @property
    def embed_dim(self) -> int:
        """Internal embedding dimension."""
        return self.num_heads * self.head_dim

    def _post_init(self):
        pass

    def _get_preproc(
        self, channels: Optional[int] = None, num_groups: Optional[int] = None
    ):
        channels = channels or self.channels
        num_groups = num_groups or self.num_groups

        preproc_layers = [AttnFlatten()]  # B x H x W x C => B x -1 x C
        if num_groups:
            preproc_layers = [
                GroupNorm(num_groups=num_groups, num_channels=channels),
                *preproc_layers,
            ]
        return nn.Sequential(*preproc_layers)

    def _get_q_preproc(self, *args, **kwargs) -> nn.Module:
        return self._get_preproc(*args, **kwargs)

    def _get_k_preproc(self, *args, **kwargs) -> nn.Module:
        return self._get_preproc(*args, **kwargs)

    def _get_v_preproc(self, *args, **kwargs) -> nn.Module:
        return self._get_preproc(*args, **kwargs)

    def _get_proy(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ) -> nn.Module:
        in_features = in_features or self.channels
        out_features = out_features or self.embed_dim
        bias = bias or self.proy_bias
        return nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )

    def _get_q_proy(self, *args, **kwargs) -> nn.Module:
        return self._get_proy(*args, **kwargs)

    def _get_k_proy(self, *args, **kwargs) -> nn.Module:
        return self._get_proy(*args, **kwargs)

    def _get_v_proy(self, *args, **kwargs) -> nn.Module:
        return self._get_proy(*args, **kwargs)

    def _q_preproc(self, q, *args, **kwargs):
        return self.preproc["q"](q, *args, **kwargs)

    def _k_preproc(self, k, *args, **kwargs):
        return self.preproc["k"](k, *args, **kwargs)

    def _v_preproc(self, v, *args, **kwargs):
        return self.preproc["v"](v, *args, **kwargs)

    def _get_out_proj(self):
        out_proj = nn.Linear(self.embed_dim, self.channels)
        if self.dropout:
            out_proj = nn.Sequential(out_proj, nn.Dropout2d(self.dropout))
        return out_proj

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        B, C, H, W = query.shape
        q = self._q_preproc(query)
        k = self._k_preproc(key)
        v = self._v_preproc(value)

        q = self.proj["q"](q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj["k"](k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj["v"](v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        result = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, **kwargs)
        result = result.transpose(1, 2).reshape(B, -1, self.embed_dim)
        result = self.out_proj(result).transpose(-1, -2).view(B, C, H, W)
        result = result + query
        return result / self.rescale_factor


class SelfVisualAttention(VisualAttention):
    """Self Visual attention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.num_groups:
            single_norm = GroupNorm(
                self.num_groups, self.channels
            )  # shared norm for the 3 inputs
            for i, p in self.preproc.items():
                self.preproc[i] = nn.Sequential(single_norm, p)

    @override
    def _get_preproc(self, *args, **kwargs):
        return AttnFlatten()

    @override
    def forward(self, query: Tensor, **kwargs) -> tuple[Tensor, Optional[Tensor]]:
        key = value = query
        return super().forward(query, key, value, **kwargs)


class CrossVisualAttention(VisualAttention):
    """Causal Visual Attention."""

    def __init__(self, context_dim: int, **kwargs) -> None:
        super().__init__(kv_channels=context_dim, **kwargs)
        # No need to preproc these two
        self.preproc.pop("k")
        self.preproc.pop("v")

    @override
    def _k_preproc(self, k, *args, **kwargs):
        """No need to reshape as it should already be embedded/flattened."""
        return k

    @override
    def _v_preproc(self, v, *args, **kwargs):
        return v

    @override
    def forward(
        self, query: Tensor, context: Tensor, **kwargs
    ) -> tuple[Tensor, Optional[Tensor]]:
        key = value = context
        return super().forward(query, key, value, **kwargs)
