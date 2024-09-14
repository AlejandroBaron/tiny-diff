from abc import ABC, abstractmethod

import torch


class DLoss(torch.nn.Module, ABC):
    """Abstract class for discriminator losses."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_real() -> torch.Tensor:
        """Loss of the real input."""

    @abstractmethod
    def forward_fake() -> torch.Tensor:
        """Loss of the fake input."""

    def forward(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss by averaging fake and real losses."""
        loss_real = self.forward_real(logits_real)
        loss_fake = self.forward_fake(logits_fake)
        d_loss = (loss_real + loss_fake) / 2
        return d_loss
