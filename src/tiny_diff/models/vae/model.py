from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
from typing_extensions import override

from tiny_diff.losses.vae_loss import VAELoss
from tiny_diff.models.vae.components import ConvDecoder, ConvEncoder
from tiny_diff.modules import ConvBlock

# Define the VAE encoder


class VAE(nn.Module, ABC):
    """Abstract class for VAE models.

    Args:
        vae_loss: loss used to train the VAE.
        encoder_cls: class of the encoder module
        decoder_cls: class of the decoder module
        transform: callable to apply to inputs
        component_kwargs: kwargs for encoder and decoder
    """

    def __init__(
        self,
        vae_loss: Optional[VAELoss] = None,
        encoder_cls: Optional[type[nn.Module]] = None,
        decoder_cls: Optional[type[nn.Module]] = None,
        transform: Optional[Callable] = None,
        component_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        component_kwargs = component_kwargs or {}
        self.encoder_cls = encoder_cls or ConvEncoder
        self.decoder_cls = decoder_cls or ConvDecoder
        self.encoder = self.encoder_cls(**component_kwargs)
        self.decoder = self.decoder_cls(**component_kwargs)
        self.transform = transform or (lambda x: x)
        self.vae_loss = vae_loss or VAELoss(recon_l=nn.L1Loss())

    def reparameterize(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Performs the reparametrization trick."""
        eps = torch.randn_like(sigma)
        sigma_clamp = torch.clamp(sigma, -20.0, 20.0)
        return mu + eps * sigma_clamp

    @abstractmethod
    def _get_mu_logsigma(self, h): ...

    @abstractmethod
    def latent_sample(self, batch_size) -> Tensor:
        """Samples from the latent space."""

    def sample(self, batch_size: int = 10, **kwargs) -> Tensor:
        """Samples from the latent space and decodes it."""
        eps = self.latent_sample(batch_size, **kwargs)
        return self.decoder(eps)

    def _encode(self, x, **kwargs):
        x_t = self.transform(x)
        return self.encoder(x_t, **kwargs)

    def forward_encode(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        """ConvEncoder side forward."""
        h = self._encode(x, **kwargs)
        mu, logsigma = self._get_mu_logsigma(h)
        sigma = torch.exp(logsigma)
        z = self.reparameterize(mu, sigma)

        return {
            "z": z,
            "mu": mu,
            "logsigma": logsigma,
            "sigma": sigma,
        }

    def forward_decode(self, z: Tensor, **kwargs) -> dict[str, Tensor]:
        """ConvDecoder side forward."""
        return {"x_hat": self.decoder(z, **kwargs)}

    @override
    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        enc_outputs = self.forward_encode(x, **kwargs)
        dec_outputs = self.forward_decode(enc_outputs["z"], **kwargs)
        outputs = {**enc_outputs, **dec_outputs}
        return outputs

    def loss_with_fwd(
        self, x, epoch: int = None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Computes the loss and forward pass."""
        fwd = self(x)
        losses = self.vae_loss(x, **fwd, epoch=epoch)
        return losses, fwd

    def loss(self, x, epoch: int = None) -> dict[str, Tensor]:
        """Computes the vae loss."""
        losses, _ = self.loss_with_fwd(x, epoch)
        return losses

    def save(self, path: str):
        """Saves the vae model."""
        return torch.save(self.state_dict(), str(path))


class ConvVAE(VAE):
    """VAE model that uses convolutional layers."""

    def __init__(
        self,
        encoder_cls: Optional[type[ConvEncoder]] = None,
        decoder_cls: Optional[type[ConvDecoder]] = None,
        **kwargs,
    ):
        super().__init__(
            encoder_cls=encoder_cls or ConvEncoder,
            decoder_cls=decoder_cls or ConvDecoder,
            **kwargs,
        )
        conv_kwargs = self.encoder.conv_kwargs(
            in_ch=self.sample_channels,
            out_ch=self.sample_channels,
            kernel_size=1,
            padding="same",
            stride=1,
            scale_factor=None,
            nonlinearity=nn.Identity(),
            attention=False,
            num_groups=None,
        )
        self.conv_mean = ConvBlock(**conv_kwargs)
        self.conv_logsigma = ConvBlock(**conv_kwargs)

    @property
    def sample_channels(self):
        """Sample size channels."""
        return self.encoder.output_channels

    @override
    def _get_mu_logsigma(self, h):
        mu, logsigma = self.conv_mean(h), self.conv_logsigma(h)
        return mu, logsigma

    @override
    def latent_sample(self, batch_size: int = 10, **kwargs):
        shape = (
            batch_size,
            self.sample_channels,
            self.encoder.embedding_size,
            self.encoder.embedding_size,
        )
        return torch.randn(shape, **kwargs)
