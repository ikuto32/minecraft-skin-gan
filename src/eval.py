from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import EvalConfig
from src.eval_io import EvalResult, append_metrics_history, save_latest_metrics
from src.eval_metrics import compute_metrics
from src.eval_plot import plot_metrics
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


def evaluate(cfg: EvalConfig) -> EvalResult:
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

    metric_values = compute_metrics(cfg=cfg, loader=loader, generator=generator, device=device)

    result = EvalResult(
        epoch=cfg.epoch if cfg.epoch is not None else int(ckpt.get("epoch", -1)) + 1,
        fid=metric_values.fid,
        kid_mean=metric_values.kid_mean,
        kid_std=metric_values.kid_std,
        sample_count=cfg.sample_count,
        seed=cfg.seed,
    )

    save_latest_metrics(result, cfg.output_dir)
    history_path = append_metrics_history(result, cfg.output_dir)
    plot_metrics(history_path, cfg.output_dir / "metrics.png")
    print(json.dumps(result.__dict__, indent=2))
    return result
