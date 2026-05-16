from pathlib import Path
import contextlib
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import EvalConfig, TrainConfig
from src.data import SkinDataset
from src.eval import evaluate
from src.losses.gan_losses import build_gan_loss
from src.models import resolve_model
from src.tracking import Tracker


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


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

    # auto profile with explicit config precedence
    auto_compile = cuda_available and has_compile and capability[0] >= 8
    auto_amp_dtype = "bfloat16" if supports_bf16 else ("float16" if cuda_available else "none")
    auto_channels_last = cuda_available

    compile_enabled = config.compile if config.compile != auto_compile else auto_compile
    amp_dtype = config.amp_dtype if config.amp_dtype != auto_amp_dtype else auto_amp_dtype
    channels_last = config.channels_last if config.channels_last != auto_channels_last else auto_channels_last
    return compile_enabled, amp_dtype, channels_last


def _update_ema(ema_model: nn.Module, model: nn.Module, beta: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema_param.lerp_(param.detach(), 1.0 - beta)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def _rand_apply(images: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return torch.ones(images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype)
    return (torch.rand(images.shape[0], 1, 1, 1, device=images.device) < p).to(images.dtype)


def _diffaugment(images: torch.Tensor, p: float, policy: str) -> torch.Tensor:
    if p <= 0.0:
        return images

    policies = {x.strip().lower() for x in policy.split(",") if x.strip()}
    out = images

    if "flip" in policies:
        apply = _rand_apply(out, p * 0.5)
        out = apply * torch.flip(out, dims=[3]) + (1.0 - apply) * out

    if "noise" in policies and p > 0.15:
        out = out + torch.randn_like(out) * (0.05 * min(1.0, p))

    if "color" in policies:
        apply = _rand_apply(out, p)
        brightness = (torch.rand(out.shape[0], 1, 1, 1, device=out.device) - 0.5) * 0.3
        contrast = 1.0 + (torch.rand(out.shape[0], 1, 1, 1, device=out.device) - 0.5) * 0.4
        colored = (out + brightness) * contrast
        out = apply * colored + (1.0 - apply) * out

    if "translation" in policies:
        apply = _rand_apply(out, p)
        max_shift = max(1, int(round(out.shape[-1] * 0.125)))
        shift_x = torch.randint(-max_shift, max_shift + 1, (out.shape[0],), device=out.device)
        shift_y = torch.randint(-max_shift, max_shift + 1, (out.shape[0],), device=out.device)
        translated = torch.stack([torch.roll(out[i], shifts=(int(shift_y[i]), int(shift_x[i])), dims=(1, 2)) for i in range(out.shape[0])], dim=0)
        out = apply * translated + (1.0 - apply) * out

    if "cutout" in policies and p > 0.2:
        cut = int(round(out.shape[-1] * 0.25))
        cut = max(2, min(out.shape[-1] - 1, cut))
        for i in range(out.shape[0]):
            if torch.rand(1, device=out.device).item() < p:
                cx = torch.randint(0, out.shape[-1], (1,), device=out.device).item()
                cy = torch.randint(0, out.shape[-2], (1,), device=out.device).item()
                x0 = max(0, cx - cut // 2)
                x1 = min(out.shape[-1], x0 + cut)
                y0 = max(0, cy - cut // 2)
                y1 = min(out.shape[-2], y0 + cut)
                out[i, :, y0:y1, x0:x1] = 0.0

    return out.clamp_(-1.0, 1.0)


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.deterministic

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
    runtime_profile = {
        "compile": compile_enabled,
        "amp_dtype": amp_dtype_name,
        "channels_last": channels_last_enabled,
        "performance_profile": config.performance_profile,
    }
    tracker = Tracker(
        config.tracking,
        {"train": config.__dict__ | {"tracking": config.tracking.__dict__}},
        runtime_profile=runtime_profile,
    )

    tracker.log_summary({
        "runtime/compile": int(compile_enabled),
        "runtime/channels_last": int(channels_last_enabled),
        "runtime/amp_dtype_is_bfloat16": int(amp_dtype_name == "bfloat16"),
        "runtime/amp_dtype_is_float16": int(amp_dtype_name == "float16"),
    })

    dataset = SkinDataset(config.data_dir)
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
    )
    dataloader_gen = torch.Generator()
    dataloader_gen.manual_seed(config.seed)
    dataloader_kwargs["generator"] = dataloader_gen
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

    loader = DataLoader(dataset, **dataloader_kwargs)

    memory_format = torch.channels_last if channels_last_enabled else torch.contiguous_format

    model_spec = resolve_model(config.model_name)
    generator = model_spec.generator_cls(z_dim=config.z_dim, **model_spec.generator_hparams).to(device, memory_format=memory_format)
    generator_ema = copy.deepcopy(generator).to(device, memory_format=memory_format)
    generator_ema.eval()
    for p in generator_ema.parameters():
        p.requires_grad_(False)

    discriminator = model_spec.discriminator_cls(**model_spec.discriminator_hparams).to(device, memory_format=memory_format)

    opt_g = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
    print(f"TTUR learning rates -> lr_g={config.lr_g:.6g}, lr_d={config.lr_d:.6g}")
    loss_strategy = build_gan_loss(
        config.loss.name,
        gp_lambda=config.loss.gp_lambda,
        r1_gamma=config.loss.r1_gamma,
        r1_interval=config.loss.r1_interval,
        r2_gamma=config.loss.r2_gamma,
        r2_interval=config.loss.r2_interval,
    )

    fixed_noise = torch.randn(64, config.z_dim, 1, 1, device=device)
    amp_enabled = device == "cuda" and amp_dtype_name != "none"
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and amp_dtype_name == "float16")
    global_step = 0
    ada_p = 0.0
    ada_sign_accum = 0.0
    ada_grad_accum = 0.0
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
            expected_model_name=config.model_name,
            expected_model_hparams={"generator": model_spec.generator_hparams, "discriminator": model_spec.discriminator_hparams},
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
                real_for_d = _diffaugment(real, ada_p, config.ada_policy) if config.use_ada else real
                if config.use_ada:
                    real_for_d = real_for_d.detach().requires_grad_(True)
                fake_for_d = _diffaugment(fake.detach(), ada_p, config.ada_policy) if config.use_ada else fake.detach()
                d_real = discriminator(real_for_d)
                d_fake = discriminator(fake_for_d)
                d_parts = loss_strategy.discriminator_loss(
                    d_real=d_real,
                    d_fake=d_fake,
                    real_images=real_for_d,
                    fake_images=fake_for_d,
                    discriminator=discriminator,
                    step=global_step,
                )
                d_loss = d_parts.loss

            if config.use_ada:
                ada_sign_accum += (d_real.detach().sign() > 0).float().sum().item()
                ada_seen += d_real.numel()
                real_grad = torch.autograd.grad(
                    outputs=d_real.sum(),
                    inputs=real_for_d,
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=False,
                )[0]
                ada_grad_accum += real_grad.detach().float().pow(2).mean().sqrt().item()
                if global_step > 0 and global_step % config.ada_interval == 0 and ada_seen > 0:
                    ada_sign = ada_sign_accum / ada_seen
                    ada_grad = ada_grad_accum / config.ada_interval
                    sign_term = (ada_sign - config.ada_target)
                    grad_term = (ada_grad - config.ada_grad_target) / max(config.ada_grad_target, 1e-8)
                    wsum = config.ada_sign_weight + config.ada_grad_weight
                    combined = (config.ada_sign_weight * sign_term + config.ada_grad_weight * grad_term) / wsum
                    ada_p = float(min(1.0, max(0.0, ada_p + (combined * config.ada_speed))))
                    ada_sign_accum = 0.0
                    ada_grad_accum = 0.0
                    ada_seen = 0


            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            generator.zero_grad(set_to_none=True)
            with autocast_context():
                fake_for_g = generator(noise)
                fake_for_g_aug = _diffaugment(fake_for_g, ada_p, config.ada_policy) if config.use_ada else fake_for_g
                output = discriminator(fake_for_g_aug)
                g_loss = loss_strategy.generator_loss(d_fake_for_g=output)

            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()
            _update_ema(generator_ema, generator, config.ema_beta)

            d_real_mean = float(d_real.mean().item())
            d_fake_mean = float(d_fake.mean().item())
            d_loss_value = float(d_loss.item())
            g_loss_value = float(g_loss.item())
            d_adv_value = float(d_parts.adv.item())
            gp_value = float(d_parts.gp.item())
            r1_penalty_value = float(d_parts.r1.item())
            r2_penalty_value = float(d_parts.r2.item())

            progress.set_postfix({
                "D_loss": f"{d_loss_value:.4f}",
                "G_loss": f"{g_loss_value:.4f}",
                "d_real": f"{d_real_mean:.4f}",
                "d_fake": f"{d_fake_mean:.4f}",
                "d_adv": f"{d_adv_value:.4f}",
                "gp": f"{gp_value:.4f}",
                "r1_penalty": f"{r1_penalty_value:.4f}",
                "r2_penalty": f"{r2_penalty_value:.4f}",
                "ada_p": f"{ada_p:.3f}",
            })
            global_step += 1
            tracker.log_metrics({
                "train/d_loss_step": d_loss_value,
                "train/g_loss_step": g_loss_value,
                "train/d_real_step": d_real_mean,
                "train/d_fake_step": d_fake_mean,
                "train/d_adv_step": d_adv_value,
                "train/gp_step": gp_value,
                "train/r1_penalty_step": r1_penalty_value,
                "train/r2_penalty_step": r2_penalty_value,
                "train/ada_p_step": ada_p,
                "train/ada_grad_step": float(ada_grad_accum / max(1, (global_step % config.ada_interval) + 1)) if config.use_ada else 0.0,
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
            "train/d_adv": d_adv_value,
            "train/gp": gp_value,
            "train/r1_penalty": r1_penalty_value,
            "train/r2_penalty": r2_penalty_value,
            "train/ada_p": ada_p,
            "train/epoch": epoch + 1,
        }, step=global_step, epoch=epoch + 1)
        tracker.log_histogram("train/fake_pixel_distribution", samples, step=global_step)
        tracker.log_image("samples", sample_path, step=global_step)
        tracker.log_artifact(sample_path, artifact_path=f"images/epoch_{epoch + 1:04d}")

        save_checkpoint(config.checkpoint_dir / "latest.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config, model_name=config.model_name, model_hparams={"generator": model_spec.generator_hparams, "discriminator": model_spec.discriminator_hparams})

        if (epoch + 1) % 10 == 0:
            save_checkpoint(config.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config, model_name=config.model_name, model_hparams={"generator": model_spec.generator_hparams, "discriminator": model_spec.discriminator_hparams})

        if config.eval_every > 0 and (epoch + 1) % config.eval_every == 0:
            eval_seeds = config.eval_seeds if config.eval_seeds else None
            eval_result = evaluate(EvalConfig(
                checkpoint=config.checkpoint_dir / "latest.pt",
                real_dir=config.data_dir,
                output_dir=config.eval_output_dir,
                sample_count=config.eval_sample_count,
                batch_size=config.eval_batch_size,
                z_dim=config.z_dim,
                seed=config.eval_seed,
                model_name=config.model_name,
                seeds=eval_seeds,
                num_workers=config.eval_num_workers,
                kid_subsets=config.kid_subsets,
                kid_subset_size=config.kid_subset_size,
                device=device,
                epoch=epoch + 1,
            ))
            metric_value = float(getattr(eval_result, config.best_metric))
            tracker.log_metrics({f"eval/{k}": float(v) for k, v in eval_result.__dict__.items() if isinstance(v, (int, float))}, epoch=epoch + 1)
            if eval_result.by_seed:
                for seed, seed_metrics in sorted(eval_result.by_seed.items()):
                    tracker.log_metrics(
                        {
                            f"eval_seed/{seed}/fid": seed_metrics["fid"],
                            f"eval_seed/{seed}/kid_mean": seed_metrics["kid_mean"],
                            f"eval_seed/{seed}/kid_std": seed_metrics["kid_std"],
                        },
                        epoch=epoch + 1,
                    )
            tracker.log_summary({
                "eval/fid_mean": eval_result.fid,
                "eval/fid_std": eval_result.fid_std,
                "eval/fid_best": eval_result.fid_best,
                "eval/kid_mean": eval_result.kid_mean,
                "eval/kid_mean_std": eval_result.kid_mean_std,
                "eval/kid_mean_best": eval_result.kid_mean_best,
            })
            tracker.log_artifact(config.eval_output_dir / "latest_metrics.json", artifact_path="eval")
            tracker.log_artifact(config.eval_output_dir / "metrics_history.csv", artifact_path="eval")
            seed_json = config.eval_output_dir / "seed_metrics_latest.json"
            seed_csv = config.eval_output_dir / "seed_metrics_latest.csv"
            if seed_json.exists():
                tracker.log_artifact(seed_json, artifact_path="eval")
            if seed_csv.exists():
                tracker.log_artifact(seed_csv, artifact_path="eval")
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                save_checkpoint(config.checkpoint_dir / "best.pt", epoch, generator, discriminator, opt_g, opt_d, generator_ema=generator_ema, best_metric=best_metric_value, train_config=config, model_name=config.model_name, model_hparams={"generator": model_spec.generator_hparams, "discriminator": model_spec.discriminator_hparams})
                print(f"New best model saved: {config.best_metric}={best_metric_value:.6f}")
            print(
                f"Eval @ epoch {epoch + 1}: "
                f"FID={eval_result.fid:.4f}, "
                f"KID={eval_result.kid_mean:.6f}±{eval_result.kid_std:.6f}"
            )

    tracker.log_summary({"best_metric": best_metric_value, "last_epoch": config.epochs})
    tracker.close()
