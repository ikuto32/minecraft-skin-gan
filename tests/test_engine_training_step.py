import contextlib
from types import SimpleNamespace

import torch

from src.engine import _train_step


class TinyGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, channels: int = 3) -> None:
        super().__init__()
        self.net = torch.nn.Conv2d(z_dim, channels, kernel_size=1, bias=False)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class TinyDiscriminator(torch.nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.net = torch.nn.Conv2d(channels, 1, kernel_size=1, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class SimpleLoss:
    def __init__(self, nan_discriminator: bool = False) -> None:
        self.nan_discriminator = nan_discriminator

    def discriminator_loss(self, d_real, d_fake, **_kwargs):
        adv = d_fake.mean() - d_real.mean()
        loss = adv
        if self.nan_discriminator:
            loss = loss * torch.tensor(float("nan"), device=loss.device)
        zero = torch.zeros((), device=loss.device)
        return SimpleNamespace(loss=loss, adv=adv, gp=zero, r1=zero, r2=zero)

    def generator_loss(self, d_fake_for_g):
        return -d_fake_for_g.mean()


def _run_step(n_critic: int, loss_strategy: SimpleLoss, step_counter: dict[str, int] | None = None):
    device = "cpu"
    z_dim = 4
    batch_size = 2
    generator = TinyGenerator(z_dim=z_dim).to(device)
    generator_ema = TinyGenerator(z_dim=z_dim).to(device)
    generator_ema.load_state_dict(generator.state_dict())
    discriminator = TinyDiscriminator().to(device)
    opt_g = torch.optim.SGD(generator.parameters(), lr=1e-2)
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=1e-2)

    if step_counter is not None:
        original_g_step = opt_g.step
        original_d_step = opt_d.step

        def counted_g_step(*args, **kwargs):
            step_counter["g"] += 1
            return original_g_step(*args, **kwargs)

        def counted_d_step(*args, **kwargs):
            step_counter["d"] += 1
            return original_d_step(*args, **kwargs)

        opt_g.step = counted_g_step  # type: ignore[method-assign]
        opt_d.step = counted_d_step  # type: ignore[method-assign]

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cfg = SimpleNamespace(n_critic=n_critic, z_dim=z_dim, ema_beta=0.0)
    real = torch.randn(batch_size, 3, 1, 1, device=device)

    return _train_step(
        real=real,
        generator=generator,
        generator_ema=generator_ema,
        discriminator=discriminator,
        opt_g=opt_g,
        opt_d=opt_d,
        loss_strategy=loss_strategy,
        scaler=scaler,
        config=cfg,
        device=device,
        global_step=0,
        ada_p=0.0,
        autocast_context=contextlib.nullcontext,
    )


def test_d_g_update_ratio_follows_n_critic():
    counts = {"g": 0, "d": 0}
    logs1, gs1 = _run_step(n_critic=1, loss_strategy=SimpleLoss(), step_counter=counts)
    assert counts == {"g": 1, "d": 1}
    assert gs1 == 1

    counts = {"g": 0, "d": 0}
    logs3, gs3 = _run_step(n_critic=3, loss_strategy=SimpleLoss(), step_counter=counts)
    assert counts == {"g": 1, "d": 3}
    assert gs3 == 3

    assert "diag/grad_norm_g" in logs1
    assert "diag/grad_norm_d" in logs1
    assert "diag/grad_norm_g" in logs3
    assert "diag/grad_norm_d" in logs3


def test_train_guard_logs_non_finite(caplog):
    with caplog.at_level("WARNING"):
        _run_step(n_critic=1, loss_strategy=SimpleLoss(nan_discriminator=True))
    assert any("[train-guard] non-finite" in rec.message for rec in caplog.records)


def test_diag_grad_norm_keys_snapshot():
    logs, _ = _run_step(n_critic=1, loss_strategy=SimpleLoss())
    diag_keys = sorted(k for k in logs if k.startswith("diag/grad_norm_"))
    assert diag_keys == ["diag/grad_norm_d", "diag/grad_norm_g"]
