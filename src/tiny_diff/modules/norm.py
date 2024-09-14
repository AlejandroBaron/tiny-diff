from torch import nn


class GroupNorm(nn.GroupNorm):
    """GroupNorm extension.

    Allows num_groups to be a negative number so it's set dynamically to the
    num_channels divided by the number of num_groups (e.g. 32//-2 = 16)
    """

    def __init__(self, num_groups: int, num_channels: int, *args, **kwargs) -> None:
        if num_groups < 0:
            num_groups = num_channels // (-num_groups)
        super().__init__(num_groups, num_channels, *args, **kwargs)
