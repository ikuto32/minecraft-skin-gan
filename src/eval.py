from __future__ import annotations

import argparse
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
    """Convert [0, 1] float RGBA/RGB tensor to uint8 RGB (N, 3, H, W)."""
    if batch.size(1) == 4:
        batch = batch[:, :3]
    batch = (batch.clamp(0, 1) * 255).round().to(torch.uint8)
    return batch


def generate_fake_batch(generator: Generator, noise: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fake = generator(noise)
    fake = (fake + 1.0) / 2.0  # [-1,1] -> [0,1]
    return fake


def evaluate(
    checkpoint: Path,
    real_dir: Path,
    output_dir: Path,
    sample_count: int,
    batch_size: int,
    z_dim: int,
    seed: int,
    num_workers: int,
    kid_subsets: int,
    kid_subset_size: int,
    device: str,
    epoch: int | None,
) -> dict[str, float | int]:
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealImageDataset(real_dir)
    if sample_count > len(dataset):
        raise ValueError(f"sample_count ({sample_count}) must be <= number of real images ({len(dataset)})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    generator = Generator(z_dim=z_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subset_size=kid_subset_size,
        subsets=kid_subsets,
        feature=2048,
        normalize=False,
    ).to(device)

    seen = 0
    for real in loader:
        if seen >= sample_count:
            break
        real = real.to(device)
        take = min(sample_count - seen, real.size(0))
        real = real[:take]

        noise = torch.randn(take, z_dim, 1, 1, device=device)
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
    kid_mean_value = float(kid_mean.item())
    kid_std_value = float(kid_std.item())

    result = {
        "epoch": epoch if epoch is not None else int(ckpt.get("epoch", -1)) + 1,
        "fid": fid_value,
        "kid_mean": kid_mean_value,
        "kid_std": kid_std_value,
        "sample_count": sample_count,
        "seed": seed,
    }

    json_path = output_dir / "latest_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    history_path = output_dir / "metrics_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "fid", "kid_mean", "kid_std", "sample_count", "seed"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    plot_metrics(history_path, output_dir / "metrics.png")

    print(json.dumps(result, indent=2))
    return result


def plot_metrics(history_csv: Path, out_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/skins"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epoch", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(
        checkpoint=args.checkpoint,
        real_dir=args.real_dir,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        seed=args.seed,
        num_workers=args.num_workers,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        device=device,
        epoch=args.epoch,
    )


if __name__ == "__main__":
    main()
