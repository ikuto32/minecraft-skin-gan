from pathlib import Path
import argparse
import random

import contextlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


class SkinDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGBA")
        return self.transform(image)


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=4, features=64):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features * 8, 4, 1, 0),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels=4, features=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),

            spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 0)),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def save_checkpoint(path, epoch, generator, discriminator, opt_g, opt_d):
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)


def load_checkpoint(path, generator, discriminator, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt["epoch"] + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/skins"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument("--r1-interval", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    dataset = SkinDataset(args.data_dir)
    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(dataset, **dataloader_kwargs)

    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format

    generator = Generator(z_dim=args.z_dim).to(device, memory_format=memory_format)
    discriminator = Discriminator().to(device, memory_format=memory_format)

    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)
    amp_enabled = device == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and args.amp_dtype == "float16")
    global_step = 0

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume,
            generator,
            discriminator,
            opt_g,
            opt_d,
            device,
        )
        print(f"Resumed from epoch {start_epoch}")

    if args.compile:
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for real in progress:
            real = real.to(device, non_blocking=(device == "cuda"))
            if args.channels_last:
                real = real.contiguous(memory_format=torch.channels_last)
            batch_size = real.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, args.z_dim, 1, 1, device=device)
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
            if args.r1_interval > 0 and global_step % args.r1_interval == 0:
                real_for_r1 = real.detach().requires_grad_(True)
                with autocast_context():
                    d_real_r1 = discriminator(real_for_r1)
                real_grad = torch.autograd.grad(
                    outputs=d_real_r1.sum(),
                    inputs=real_for_r1,
                    create_graph=True,
                )[0]
                r1_penalty = real_grad.pow(2).flatten(1).sum(1).mean()
                d_loss = d_loss + 0.5 * args.r1_gamma * r1_penalty

            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            # Train Generator
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

        save_checkpoint(
            Path("checkpoints") / "latest.pt",
            epoch,
            generator,
            discriminator,
            opt_g,
            opt_d,
        )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                Path("checkpoints") / f"epoch_{epoch + 1:04d}.pt",
                epoch,
                generator,
                discriminator,
                opt_g,
                opt_d,
            )


if __name__ == "__main__":
    main()
