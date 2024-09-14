import torch
from typing_extensions import override

from tiny_diff.losses.discriminator.d_loss import DLoss


class HingeDLoss(DLoss):
    """Hinge loss for discriminator/GAN tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_fake(self, logits_fake: torch.Tensor):
        loss_fake = torch.mean(torch.nn.functional.relu(1.0 + logits_fake))
        return loss_fake.mean()

    @override
    def forward_real(self, logits_real: torch.Tensor):
        loss_real = torch.mean(torch.nn.functional.relu(1.0 - logits_real))
        return loss_real.mean()
