from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

from src.config import EvalConfig
from src.models import Generator


class RealImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx]).convert("RGBA")
        return self.transform(image)


def to_uint8_rgb(batch: torch.Tensor) -> torch.Tensor:
    if batch.size(1) == 4:
        batch = batch[:, :3]
    return (batch.clamp(0, 1) * 255).round().to(torch.uint8)


def generate_fake_batch(generator: Generator, noise: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fake = generator(noise)
    return (fake + 1.0) / 2.0


def evaluate(cfg: EvalConfig) -> dict[str, float | int]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealImageDataset(cfg.real_dir)
    if cfg.sample_count > len(dataset):
        raise ValueError(f"sample_count ({cfg.sample_count}) must be <= number of real images ({len(dataset)})")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    generator = Generator(z_dim=cfg.z_dim).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subset_size=cfg.kid_subset_size,
        subsets=cfg.kid_subsets,
        feature=2048,
        normalize=False,
    ).to(device)

    seen = 0
    for real in loader:
        if seen >= cfg.sample_count:
            break
        real = real.to(device)
        take = min(cfg.sample_count - seen, real.size(0))
        real = real[:take]

        noise = torch.randn(take, cfg.z_dim, 1, 1, device=device)
        fake = generate_fake_batch(generator, noise)

        real_u8 = to_uint8_rgb(real)
        fake_u8 = to_uint8_rgb(fake)

        fid.update(real_u8, real=True)
        fid.update(fake_u8, real=False)
        kid.update(real_u8, real=True)
        kid.update(fake_u8, real=False)
        seen += take

    fid_value = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()

    result = {
        "epoch": cfg.epoch if cfg.epoch is not None else int(ckpt.get("epoch", -1)) + 1,
        "fid": fid_value,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "sample_count": cfg.sample_count,
        "seed": cfg.seed,
    }

    json_path = cfg.output_dir / "latest_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    history_path = cfg.output_dir / "metrics_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "fid", "kid_mean", "kid_std", "sample_count", "seed"])
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    plot_metrics(history_path, cfg.output_dir / "metrics.png")
    print(json.dumps(result, indent=2))
    return result


def plot_metrics(history_csv: Path, out_path: Path) -> None:
    with history_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    epochs = [int(r["epoch"]) for r in rows]
    fids = [float(r["fid"]) for r in rows]
    kid_means = [float(r["kid_mean"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, fids, marker="o")
    axes[0].set_title("FID")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, kid_means, marker="o", color="tab:orange")
    axes[1].set_title("KID mean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Lower is better")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
