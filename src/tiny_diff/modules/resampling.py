from typing import Literal

from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import override


class Interpolation(nn.Module):
    """Layer used to manipulate input's tensor size using torch.functional.interpolate.

    Args:
        scale_factor (float): Scaling factor for downscaling.
        mode (str): Interpolation mode. Options include, etc.
    """

    def __init__(
        self,
        scale_factor: float = 2.0,
        mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ] = "bilinear",
        **kwargs,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.kwargs = kwargs

    @override
    def forward(self, x: Tensor) -> Tensor:
        x_downscale = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, **self.kwargs
        )
        return x_downscale


class Upsample(Interpolation):
    """Interpolation layer to increase input's size."""

    def __init__(self, scale_factor: float = 2.0, **kwargs):
        scale_factor = scale_factor if scale_factor > 1 else 1 / scale_factor
        super().__init__(scale_factor=scale_factor, **kwargs)


class Downsample(Interpolation):
    """Interpolation layer to reduce input's size."""

    def __init__(self, scale_factor: float = 2.0, **kwargs):
        scale_factor = 1 / scale_factor if scale_factor > 1 else scale_factor
        super().__init__(scale_factor=scale_factor, **kwargs)
