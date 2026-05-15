from __future__ import annotations

import json
import math
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def evaluate(cfg: EvalConfig) -> EvalResult:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.deterministic

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealImageDataset(cfg.real_dir)
    if cfg.sample_count > len(dataset):
        raise ValueError(f"sample_count ({cfg.sample_count}) must be <= number of real images ({len(dataset)})")

    dataloader_gen = torch.Generator()
    dataloader_gen.manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
        generator=dataloader_gen,
    )

    generator = Generator(z_dim=cfg.z_dim).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    seeds = cfg.seeds or [cfg.seed]
    metrics_by_seed: dict[int, dict[str, float]] = {}
    fid_values: list[float] = []
    kid_mean_values: list[float] = []
    kid_std_values: list[float] = []

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        seed_cfg = EvalConfig(**(cfg.__dict__ | {"seed": seed, "seeds": None}))
        metric_values = compute_metrics(cfg=seed_cfg, loader=loader, generator=generator, device=device)
        fid_values.append(metric_values.fid)
        kid_mean_values.append(metric_values.kid_mean)
        kid_std_values.append(metric_values.kid_std)
        metrics_by_seed[seed] = {
            "fid": metric_values.fid,
            "kid_mean": metric_values.kid_mean,
            "kid_std": metric_values.kid_std,
        }

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    def _std(values: list[float], mean: float) -> float:
        if len(values) <= 1:
            return 0.0
        return float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))

    fid_mean = _mean(fid_values)
    kid_mean_mean = _mean(kid_mean_values)
    kid_std_mean = _mean(kid_std_values)

    result = EvalResult(
        epoch=cfg.epoch if cfg.epoch is not None else int(ckpt.get("epoch", -1)) + 1,
        fid=fid_mean,
        fid_std=_std(fid_values, fid_mean),
        fid_best=min(fid_values),
        fid_worst=max(fid_values),
        kid_mean=kid_mean_mean,
        kid_std=kid_std_mean,
        kid_mean_std=_std(kid_mean_values, kid_mean_mean),
        kid_mean_best=min(kid_mean_values),
        kid_mean_worst=max(kid_mean_values),
        sample_count=cfg.sample_count,
        seed=seeds[0],
        seeds=seeds,
    )

    save_latest_metrics(result, cfg.output_dir)
    history_path = append_metrics_history(result, cfg.output_dir)
    plot_metrics(history_path, cfg.output_dir / "metrics.png")
    print(json.dumps({"summary": result.__dict__, "by_seed": metrics_by_seed}, indent=2))
    return result
