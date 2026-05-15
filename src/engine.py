from pathlib import Path
import contextlib
import copy
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import EvalConfig, TrainConfig
from src.data import SkinDataset
from src.eval import evaluate
from src.models import Discriminator, Generator
from src.tracking import Tracker


def _select_runtime_profile(config: TrainConfig, device: str) -> tuple[bool, str, bool]:
    if config.performance_profile == "safe":
        return False, "none", False

    cuda_available = device == "cuda"
    has_compile = hasattr(torch, "compile")
    capability = torch.cuda.get_device_capability() if cuda_available else (0, 0)
    supports_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    if config.performance_profile == "max":
        compile_enabled = cuda_available and has_compile
        amp_dtype = "bfloat16" if supports_bf16 else ("float16" if cuda_available else "none")
        channels_last = cuda_available
        return compile_enabled, amp_dtype, channels_last

    # auto profile
    compile_enabled = cuda_available and has_compile and capability[0] >= 8
    amp_dtype = "bfloat16" if supports_bf16 else ("float16" if cuda_available else "none")
    channels_last = cuda_available
    return compile_enabled, amp_dtype, channels_last


def _update_ema(ema_model: Generator, model: Generator, beta: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema_param.lerp_(param.detach(), 1.0 - beta)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def _augment(images: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return images

    batch = images.shape[0]
    device = images.device

    flip_mask = (torch.rand(batch, 1, 1, 1, device=device) < (p * 0.5)).to(images.dtype)
    flipped = torch.flip(images, dims=[3])
    images = flip_mask * flipped + (1.0 - flip_mask) * images

    if p > 0.25:
        noise_std = 0.05 * min(1.0, p)
        images = images + torch.randn_like(images) * noise_std

    return images.clamp_(-1.0, 1.0)


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    compile_enabled, amp_dtype_name, channels_last_enabled = _select_runtime_profile(config, device)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name()
        capability = torch.cuda.get_device_capability()
        print(f"GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]})")
    else:
        print("GPU: not available (CPU runtime)")
    print(
        "Performance profile="
        f"{config.performance_profile} -> compile={compile_enabled}, "
        f"amp_dtype={amp_dtype_name}, channels_last={channels_last_enabled}"
    )

    Path("outputs").mkdir(exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(config.tracking, {"train": config.__dict__ | {"tracking": config.tracking.__dict__}})

    dataset = SkinDataset(config.data_dir)
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

    loader = DataLoader(dataset, **dataloader_kwargs)

    memory_format = torch.channels_last if channels_last_enabled else torch.contiguous_format

    generator = Generator(z_dim=config.z_dim).to(device, memory_format=memory_format)
    generator_ema = copy.deepcopy(generator).to(device, memory_format=memory_format)
    generator_ema.eval()
    for p in generator_ema.parameters():
        p.requires_grad_(False)

    discriminator = Discriminator().to(device, memory_format=memory_format)

    opt_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, config.z_dim, 1, 1, device=device)
    amp_enabled = device == "cuda" and amp_dtype_name != "none"
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and amp_dtype_name == "float16")
    global_step = 0
    ada_p = 0.0
    ada_sign_accum = 0.0
    ada_seen = 0

    start_epoch = 0
    best_metric_value = float("inf")
    resume_path = config.resume
    latest_path = config.checkpoint_dir / "latest.pt"
    if resume_path is None and config.auto_resume and latest_path.exists():
        resume_path = latest_path

    if resume_path:
        resume_info = load_checkpoint(
            resume_path,
            generator,
            discriminator,
            opt_g,
            opt_d,
            device,
            generator_ema=generator_ema,
        )
        start_epoch = resume_info["start_epoch"]
        best_metric_value = resume_info["best_metric"]
        _update_ema(generator_ema, generator, beta=0.0)
        print(
            f"Resumed from {resume_path} at epoch {start_epoch} "
            f"(schema_version={resume_info['schema_version']}, model_type={resume_info['model_type']})"
        )

    if compile_enabled:
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)

    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for real in progress:
            real = real.to(device, non_blocking=(device == "cuda"))
            if channels_last_enabled:
                real = real.contiguous(memory_format=torch.channels_last)
            batch_size = real.size(0)

            noise = torch.randn(batch_size, config.z_dim, 1, 1, device=device)

            def autocast_context():
                if amp_enabled:
                    return torch.autocast(device_type="cuda", dtype=amp_dtype)
                return contextlib.nullcontext()

            discriminator.zero_grad(set_to_none=True)
            with autocast_context():
                fake = generator(noise)
                real_for_d = _augment(real, ada_p) if config.use_ada else real
                fake_for_d = _augment(fake.detach(), ada_p) if config.use_ada else fake.detach()
                d_real = discriminator(real_for_d)
                d_fake = discriminator(fake_for_d)
                real_loss = torch.relu(1.0 - d_real).mean()
                fake_loss = torch.relu(1.0 + d_fake).mean()
                d_loss = real_loss + fake_loss

            if config.use_ada:
                ada_sign_accum += (d_real.detach().sign() > 0).float().sum().item()
                ada_seen += d_real.numel()
                if global_step > 0 and global_step % config.ada_interval == 0 and ada_seen > 0:
                    ada_sign = ada_sign_accum / ada_seen
                    adjust = (ada_sign - config.ada_target) * config.ada_speed
                    ada_p = float(min(1.0, max(0.0, ada_p + adjust)))
                    ada_sign_accum = 0.0
                    ada_seen = 0

            r1_penalty = torch.tensor(0.0, device=device)
            if config.r1_interval > 0 and global_step % config.r1_interval == 0:
                real_for_r1 = real.detach().requires_grad_(True)
                with autocast_context():
                    d_real_r1 = discriminator(real_for_r1)
                real_grad = torch.autograd.grad(
                    outputs=d_real_r1.sum(),
                    inputs=real_for_r1,
                    create_graph=True,
                )[0]
                r1_penalty = real_grad.pow(2).flatten(1).sum(1).mean()
                d_loss = d_loss + 0.5 * config.r1_gamma * r1_penalty

            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            generator.zero_grad(set_to_none=True)
            with autocast_context():
                fake_for_g = generator(noise)
                fake_for_g_aug = _augment(fake_for_g, ada_p) if config.use_ada else fake_for_g
                output = discriminator(fake_for_g_aug)
                g_loss = -output.mean()

            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()
            _update_ema(generator_ema, generator, config.ema_beta)

            d_real_mean = float(d_real.mean().item())
            d_fake_mean = float(d_fake.mean().item())
            d_loss_value = float(d_loss.item())
            g_loss_value = float(g_loss.item())
            r1_penalty_value = float(r1_penalty.item())

            progress.set_postfix({
                "D_loss": f"{d_loss_value:.4f}",
                "G_loss": f"{g_loss_value:.4f}",
                "d_real": f"{d_real_mean:.4f}",
                "d_fake": f"{d_fake_mean:.4f}",
                "r1_penalty": f"{r1_penalty_value:.4f}",
                "ada_p": f"{ada_p:.3f}",
            })
            global_step += 1
            tracker.log_metrics({
                "train/d_loss_step": d_loss_value,
                "train/g_loss_step": g_loss_value,
                "train/d_real_step": d_real_mean,
                "train/d_fake_step": d_fake_mean,
                "train/r1_penalty_step": r1_penalty_value,
                "train/ada_p_step": ada_p,
                "train/lr_g": float(opt_g.param_groups[0]["lr"]),
                "train/lr_d": float(opt_d.param_groups[0]["lr"]),
            }, step=global_step)

        with torch.no_grad():
            samples = generator_ema(fixed_noise).detach().cpu()
            samples = (samples + 1) / 2
            sample_path = Path(f"outputs/epoch_{epoch + 1:04d}.png")
            save_image(samples, sample_path, nrow=8)

        tracker.log_metrics({
            "train/d_loss": d_loss_value,
            "train/g_loss": g_loss_value,
            "train/d_real": d_real_mean,
            "train/d_fake": d_fake_mean,
            "train/r1_penalty": r1_penalty_value,
            "train/ada_p": ada_p,
            "train/epoch": epoch + 1,
        }, step=global_step, epoch=epoch + 1)
        tracker.log_histogram("train/fake_pixel_distribution", samples, step=global_step)
        tracker.log_image("samples", sample_path, step=global_step)
        tracker.log_artifact(sample_path, artifact_path=f"images/epoch_{epoch + 1:04d}")

        save_checkpoint(config.checkpoint_dir / "latest.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(config.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config)

        if config.eval_every > 0 and (epoch + 1) % config.eval_every == 0:
            eval_result = evaluate(EvalConfig(
                checkpoint=config.checkpoint_dir / "latest.pt",
                real_dir=config.data_dir,
                output_dir=config.eval_output_dir,
                sample_count=config.eval_sample_count,
                batch_size=config.eval_batch_size,
                z_dim=config.z_dim,
                seed=config.eval_seed,
                seeds=config.eval_seeds,
                num_workers=config.eval_num_workers,
                kid_subsets=config.kid_subsets,
                kid_subset_size=config.kid_subset_size,
                device=device,
                epoch=epoch + 1,
            ))
            metric_value = float(getattr(eval_result, config.best_metric))
            tracker.log_metrics({f"eval/{k}": float(v) for k, v in eval_result.__dict__.items() if isinstance(v, (int, float))}, epoch=epoch + 1)
            tracker.log_artifact(config.eval_output_dir / "latest_metrics.json", artifact_path="eval")
            tracker.log_artifact(config.eval_output_dir / "metrics_history.csv", artifact_path="eval")
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                save_checkpoint(config.checkpoint_dir / "best.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config)
                print(f"New best model saved: {config.best_metric}={best_metric_value:.6f}")
            print(
                f"Eval @ epoch {epoch + 1}: "
                f"FID={eval_result.fid:.4f}, "
                f"KID={eval_result.kid_mean:.6f}±{eval_result.kid_std:.6f}"
            )

    tracker.log_summary({"best_metric": best_metric_value, "last_epoch": config.epochs})
    tracker.close()
