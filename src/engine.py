from pathlib import Path
import contextlib
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import TrainConfig
from src.data import SkinDataset
from src.eval import evaluate
from src.models import Discriminator, Generator


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

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

    memory_format = torch.channels_last if config.channels_last else torch.contiguous_format

    generator = Generator(z_dim=config.z_dim).to(device, memory_format=memory_format)
    discriminator = Discriminator().to(device, memory_format=memory_format)

    opt_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, config.z_dim, 1, 1, device=device)
    amp_enabled = device == "cuda" and config.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.amp_dtype == "float16")
    global_step = 0

    start_epoch = 0
    if config.resume:
        start_epoch = load_checkpoint(
            config.resume,
            generator,
            discriminator,
            opt_g,
            opt_d,
            device,
        )
        print(f"Resumed from epoch {start_epoch}")

    if config.compile:
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)

    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for real in progress:
            real = real.to(device, non_blocking=(device == "cuda"))
            if config.channels_last:
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
                d_real = discriminator(real)
                d_fake = discriminator(fake.detach())
                real_loss = torch.relu(1.0 - d_real).mean()
                fake_loss = torch.relu(1.0 + d_fake).mean()
                d_loss = real_loss + fake_loss

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
                output = discriminator(fake_for_g)
                g_loss = -output.mean()

            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            progress.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "d_real": f"{d_real.mean().item():.4f}",
                "d_fake": f"{d_fake.mean().item():.4f}",
                "r1_penalty": f"{r1_penalty.item():.4f}",
            })
            global_step += 1

        with torch.no_grad():
            samples = generator(fixed_noise).detach().cpu()
            samples = (samples + 1) / 2
            save_image(samples, f"outputs/epoch_{epoch + 1:04d}.png", nrow=8)

        save_checkpoint(Path("checkpoints") / "latest.pt", epoch, generator, discriminator, opt_g, opt_d)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(Path("checkpoints") / f"epoch_{epoch + 1:04d}.pt", epoch, generator, discriminator, opt_g, opt_d)

        if config.eval_every > 0 and (epoch + 1) % config.eval_every == 0:
            eval_result = evaluate(
                checkpoint=Path("checkpoints") / "latest.pt",
                real_dir=config.data_dir,
                output_dir=config.eval_output_dir,
                sample_count=config.eval_sample_count,
                batch_size=config.eval_batch_size,
                z_dim=config.z_dim,
                seed=config.eval_seed,
                num_workers=config.eval_num_workers,
                kid_subsets=config.kid_subsets,
                kid_subset_size=config.kid_subset_size,
                device=device,
                epoch=epoch + 1,
            )
            print(
                f"Eval @ epoch {epoch + 1}: "
                f"FID={eval_result['fid']:.4f}, "
                f"KID={eval_result['kid_mean']:.6f}±{eval_result['kid_std']:.6f}"
            )
