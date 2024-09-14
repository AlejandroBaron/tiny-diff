import torch
from typing_extensions import override

from tiny_diff.losses.discriminator.d_loss import DLoss


class VanillaDLoss(DLoss):
    """Discriminator loss used in the original GAN paper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_fake(self, logits_fake):
        return torch.mean(torch.nn.functional.softplus(logits_fake)).mean()

    @override
    def forward_real(self, logits_real):
        return torch.mean(torch.nn.functional.softplus(-logits_real)).mean()
