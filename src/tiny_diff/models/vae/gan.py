from typing import Literal

from torch import Tensor, autograd, clamp

from tiny_diff.models.discriminator import Pix2PixDiscriminator
from tiny_diff.models.vae.model import VAE


class VAEGAN:
    """VAE + GAN.

    Model that coordinates a vae and a discriminator to increase reconstruction quality.

    Args:
        vae: VAE model.
        discriminator: Discriminator in the GAN pair.
        disc_weight: discriminator weight for the discriminator loss term.
        gen_warmup: last epoch in which the generator will be trained alone.
    """

    def __init__(
        self,
        vae: VAE,
        discriminator: Pix2PixDiscriminator,
        disc_weight: float = 1.0,
        gen_warmup: int = 60,
    ):
        self.vae = vae
        self.discriminator = discriminator
        self.train_generator = True
        self.disc_weight = disc_weight
        self.gen_warmup = gen_warmup

    def disc_loss_weight(self, loss, epoch) -> float:
        """Discriminator loss term weight."""
        return self.disc_weight

    def gen_loss(self, x, epoch: int = None) -> dict[str, Tensor]:
        """Generator loss."""
        loss, fwd = self.vae.loss_with_fwd(x, epoch)
        if epoch >= self.gen_warmup:
            reconstructions = fwd["x_hat"]
            loss["gen_disc"] = self.discriminator(reconstructions).mean()
            loss["loss"] -= loss["gen_disc"] * self.disc_loss_weight(loss, epoch)
        return loss

    def loss(
        self,
        x,
        epoch: int = None,
        mode: Literal["generator", "discriminator"] = "generator",
    ):
        """Computes the VAEGAN loss."""
        if mode == "generator":
            return self.gen_loss(x, epoch)
        if mode == "discriminator":
            return self.discriminator.loss(x, self.vae(x)["x_hat"])
        raise ValueError("mode not valid. Use either 'generator' or 'discriminator'.")


class VAEGANAdapt(VAEGAN):
    """VAEGAN with an adaptative weight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_lambda = 1

    def adaptative_weight(self, recon_l, disc_l, layer, delta=1e-4) -> float:
        """Adaptative weight used in Taming Transformers paper."""
        if recon_l.requires_grad:
            recon_grads = autograd.grad(recon_l, layer.weight, retain_graph=True)[
                0
            ].norm()
            disc_grads = autograd.grad(disc_l, layer.weight, retain_graph=True)[
                0
            ].norm()
            adp_w = recon_grads / disc_grads + delta
            lambd = clamp(adp_w, 0.0, 1e4).detach()
            self.last_lambda = lambd
            return lambd
        return self.last_lambda

    def disc_loss_weight(self, loss: dict[str, Tensor], epoch: int) -> float:
        """Value used to weight the gan loss."""
        lambd = self.adaptative_weight(
            recon_l=loss["recon_loss"],
            disc_l=loss["gen_disc"],
            layer=self.vae.decoder.last_layer,
        )
        return lambd * self.disc_weight
