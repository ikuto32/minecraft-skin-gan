from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class DiscriminatorLossOutput:
    loss: torch.Tensor
    adv: torch.Tensor
    gp: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor


class GanLossStrategy:
    def discriminator_loss(
        self,
        *,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        discriminator: torch.nn.Module,
        step: int,
    ) -> DiscriminatorLossOutput:
        raise NotImplementedError

    def generator_loss(self, *, d_fake_for_g: torch.Tensor, d_real_for_g: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class HingeLossStrategy(GanLossStrategy):
    def discriminator_loss(self, *, d_real: torch.Tensor, d_fake: torch.Tensor, real_images: torch.Tensor, fake_images: torch.Tensor, discriminator: torch.nn.Module, step: int) -> DiscriminatorLossOutput:
        del real_images, fake_images, discriminator, step
        real_loss = F.relu(1.0 - d_real).mean()
        fake_loss = F.relu(1.0 + d_fake).mean()
        adv = real_loss + fake_loss
        zero = torch.zeros((), device=adv.device, dtype=adv.dtype)
        return DiscriminatorLossOutput(loss=adv, adv=adv, gp=zero, r1=zero, r2=zero)

    def generator_loss(self, *, d_fake_for_g: torch.Tensor, d_real_for_g: torch.Tensor | None = None) -> torch.Tensor:
        del d_real_for_g
        return -d_fake_for_g.mean()


class WganGpLossStrategy(GanLossStrategy):
    def __init__(self, gp_lambda: float) -> None:
        self.gp_lambda = gp_lambda

    def discriminator_loss(self, *, d_real: torch.Tensor, d_fake: torch.Tensor, real_images: torch.Tensor, fake_images: torch.Tensor, discriminator: torch.nn.Module, step: int) -> DiscriminatorLossOutput:
        del step
        adv = d_fake.mean() - d_real.mean()
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device, dtype=real_images.dtype)
        interpolates = (alpha * real_images + (1.0 - alpha) * fake_images).detach().requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        grads = torch.autograd.grad(outputs=d_interpolates.sum(), inputs=interpolates, create_graph=True)[0]
        gp = ((grads.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
        loss = adv + self.gp_lambda * gp
        zero = torch.zeros((), device=adv.device, dtype=adv.dtype)
        return DiscriminatorLossOutput(loss=loss, adv=adv, gp=gp, r1=zero, r2=zero)

    def generator_loss(self, *, d_fake_for_g: torch.Tensor, d_real_for_g: torch.Tensor | None = None) -> torch.Tensor:
        del d_real_for_g
        return -d_fake_for_g.mean()


class LogisticR1R2LossStrategy(GanLossStrategy):
    def __init__(self, r1_gamma: float, r1_interval: int, r2_gamma: float, r2_interval: int) -> None:
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval
        self.r2_gamma = r2_gamma
        self.r2_interval = r2_interval

    def discriminator_loss(self, *, d_real: torch.Tensor, d_fake: torch.Tensor, real_images: torch.Tensor, fake_images: torch.Tensor, discriminator: torch.nn.Module, step: int) -> DiscriminatorLossOutput:
        adv = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
        zero = torch.zeros((), device=adv.device, dtype=adv.dtype)
        r1 = zero
        r2 = zero
        loss = adv
        if self.r1_interval > 0 and step % self.r1_interval == 0:
            real_for_r1 = real_images.detach().requires_grad_(True)
            d_real_r1 = discriminator(real_for_r1)
            real_grad = torch.autograd.grad(outputs=d_real_r1.sum(), inputs=real_for_r1, create_graph=True)[0]
            r1 = real_grad.pow(2).flatten(1).sum(1).mean()
            loss = loss + 0.5 * self.r1_gamma * r1
        if self.r2_gamma > 0.0 and self.r2_interval > 0 and step % self.r2_interval == 0:
            fake_for_r2 = fake_images.detach().requires_grad_(True)
            d_fake_r2 = discriminator(fake_for_r2)
            fake_grad = torch.autograd.grad(outputs=d_fake_r2.sum(), inputs=fake_for_r2, create_graph=True)[0]
            r2 = fake_grad.pow(2).flatten(1).sum(1).mean()
            loss = loss + 0.5 * self.r2_gamma * r2
        return DiscriminatorLossOutput(loss=loss, adv=adv, gp=zero, r1=r1, r2=r2)

    def generator_loss(self, *, d_fake_for_g: torch.Tensor, d_real_for_g: torch.Tensor | None = None) -> torch.Tensor:
        del d_real_for_g
        return F.softplus(-d_fake_for_g).mean()


class R3GanRelativisticLossStrategy(GanLossStrategy):
    def __init__(self, rel_scale: float, rel_margin: float, r1_gamma: float, r1_interval: int, r2_gamma: float, r2_interval: int) -> None:
        self.rel_scale = rel_scale
        self.rel_margin = rel_margin
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval
        self.r2_gamma = r2_gamma
        self.r2_interval = r2_interval

    def discriminator_loss(self, *, d_real: torch.Tensor, d_fake: torch.Tensor, real_images: torch.Tensor, fake_images: torch.Tensor, discriminator: torch.nn.Module, step: int) -> DiscriminatorLossOutput:
        rel = self.rel_scale * (d_real - d_fake - self.rel_margin)
        adv = F.softplus(-rel).mean()
        zero = torch.zeros((), device=adv.device, dtype=adv.dtype)
        r1 = zero
        r2 = zero
        loss = adv
        if self.r1_gamma > 0.0 and self.r1_interval > 0 and step % self.r1_interval == 0:
            real_for_r1 = real_images.detach().requires_grad_(True)
            d_real_r1 = discriminator(real_for_r1)
            real_grad = torch.autograd.grad(outputs=d_real_r1.sum(), inputs=real_for_r1, create_graph=True)[0]
            r1 = real_grad.pow(2).flatten(1).sum(1).mean()
            loss = loss + 0.5 * self.r1_gamma * r1
        if self.r2_gamma > 0.0 and self.r2_interval > 0 and step % self.r2_interval == 0:
            fake_for_r2 = fake_images.detach().requires_grad_(True)
            d_fake_r2 = discriminator(fake_for_r2)
            fake_grad = torch.autograd.grad(outputs=d_fake_r2.sum(), inputs=fake_for_r2, create_graph=True)[0]
            r2 = fake_grad.pow(2).flatten(1).sum(1).mean()
            loss = loss + 0.5 * self.r2_gamma * r2
        return DiscriminatorLossOutput(loss=loss, adv=adv, gp=zero, r1=r1, r2=r2)

    def generator_loss(self, *, d_fake_for_g: torch.Tensor, d_real_for_g: torch.Tensor | None = None) -> torch.Tensor:
        if d_real_for_g is None:
            return F.softplus(-self.rel_scale * (d_fake_for_g - self.rel_margin)).mean()
        rel = self.rel_scale * (d_fake_for_g - d_real_for_g - self.rel_margin)
        return F.softplus(-rel).mean()


def build_gan_loss(name: Literal["hinge", "wgan_gp", "logistic_r1", "r3gan_relativistic"], *, gp_lambda: float, r1_gamma: float, r1_interval: int, r2_gamma: float, r2_interval: int, rel_scale: float, rel_margin: float) -> GanLossStrategy:
    if name == "hinge":
        return HingeLossStrategy()
    if name == "wgan_gp":
        return WganGpLossStrategy(gp_lambda=gp_lambda)
    if name == "r3gan_relativistic":
        return R3GanRelativisticLossStrategy(rel_scale=rel_scale, rel_margin=rel_margin, r1_gamma=r1_gamma, r1_interval=r1_interval, r2_gamma=r2_gamma, r2_interval=r2_interval)
    return LogisticR1R2LossStrategy(r1_gamma=r1_gamma, r1_interval=r1_interval, r2_gamma=r2_gamma, r2_interval=r2_interval)
