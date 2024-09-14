import math
from typing import Callable, Optional

from torch import Tensor, mean, nn
from typing_extensions import override


class BetaPolicy(Callable):
    """Function representing a Beta Policy.

    This sigmoid like function is used to reflect a weight that changes over each epoch.

    Args:
        a: minimum value
        b: maximum value
        offset: location of sigmoid's middle
        slope: rate of growth. the higher the sharper is the growth.
    """

    def __init__(
        self, a: float = 1e-3, b: float = 1e-2, offset: float = 60, slope: float = 5
    ):
        self.a = a
        self.b = b
        self.offset = offset
        self.slope = slope

    def __call__(self, x: float) -> float:
        """Returns the Beta policy value."""
        s = self.a + (self.b - self.a) / (1 + math.exp(-(x - self.offset) / self.slope))
        return s


class VAELoss(nn.Module):
    """Loss for VAE models.

    Args:
        recon_l: reconstruction loss. Will be added to the KL divergence
            against a 0,1 Normal
        kl_w_policy: KL divergence weight. Must be a function of epoch
        kl_weight: constant weight, as opposed to using a policy
    """

    def __init__(
        self,
        recon_l: Optional[nn.Module] = None,
        kl_w_policy: Optional[Callable] = None,
        kl_weight: float = 1,
    ):
        super().__init__()
        self.recon_l = recon_l or nn.L1Loss()
        self.kl_w_policy = kl_w_policy or (lambda epoch: kl_weight)

    def _kl_divergence(self, mu: Tensor, logsigma: Tensor, sigma: Tensor):
        return -0.5 * mean(1 + 2 * logsigma - mu.pow(2) - sigma.pow(2))

    @override
    def forward(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        logsigma: Tensor,
        sigma: Tensor,
        epoch: Optional[int] = None,
        **kwargs,
    ):
        kl_weight = self.kl_w_policy(epoch) if epoch else 1
        recon_loss = self.recon_l(x_hat, x)
        kl_divergence = self._kl_divergence(mu, logsigma, sigma)
        return {
            "loss": recon_loss + kl_divergence * kl_weight,
            "recon_loss": recon_loss,
            "kl_divergence": kl_divergence,
        }
