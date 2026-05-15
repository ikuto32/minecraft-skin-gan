from pathlib import Path
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
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
            nn.Conv2d(channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),

            nn.Conv2d(features * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    dataset = SkinDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    generator = Generator(z_dim=args.z_dim).to(device)
    discriminator = Discriminator().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

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

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for real in progress:
            real = real.to(device)
            batch_size = real.size(0)

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # Train Discriminator
            noise = torch.randn(batch_size, args.z_dim, 1, 1, device=device)
            fake = generator(noise)

            discriminator.zero_grad()

            real_loss = criterion(discriminator(real), real_labels)
            fake_loss = criterion(discriminator(fake.detach()), fake_labels)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            opt_d.step()

            # Train Generator
            generator.zero_grad()

            output = discriminator(fake)
            g_loss = criterion(output, real_labels)

            g_loss.backward()
            opt_g.step()

            progress.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
            })

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