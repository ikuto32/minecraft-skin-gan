from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalResult:
    epoch: int
    fid: float
    kid_mean: float
    kid_std: float
    sample_count: int
    seed: int
    fid_std: float = 0.0
    fid_best: float = 0.0
    fid_worst: float = 0.0
    kid_mean_std: float = 0.0
    kid_mean_best: float = 0.0
    kid_mean_worst: float = 0.0
    precision: float = 0.0
    precision_std: float = 0.0
    precision_best: float = 0.0
    precision_worst: float = 0.0
    recall: float = 0.0
    recall_std: float = 0.0
    recall_best: float = 0.0
    recall_worst: float = 0.0
    seeds: list[int] | None = None
    by_seed: dict[int, dict[str, float]] | None = None


def save_latest_metrics(result: EvalResult, output_dir: Path) -> Path:
    json_path = output_dir / "latest_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    return json_path


def append_metrics_history(result: EvalResult, output_dir: Path) -> Path:
    history_path = output_dir / "metrics_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
                "epoch",
                "fid",
                "fid_std",
                "fid_best",
                "fid_worst",
                "kid_mean",
                "kid_std",
                "kid_mean_std",
                "kid_mean_best",
                "kid_mean_worst",
                "sample_count",
                "seed",
                "seeds",
                "precision",
                "precision_std",
                "precision_best",
                "precision_worst",
                "recall",
                "recall_std",
                "recall_best",
                "recall_worst",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        
        if write_header:
            writer.writeheader()
        row = asdict(result)
        row["seeds"] = ",".join(str(s) for s in (result.seeds or [result.seed]))
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return history_path


def save_seed_metrics(result: EvalResult, output_dir: Path) -> tuple[Path, Path] | None:
    if not result.by_seed:
        return None

    json_path = output_dir / "seed_metrics_latest.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": result.epoch,
                "summary": {
                    "fid": result.fid,
                    "fid_std": result.fid_std,
                    "fid_best": result.fid_best,
                    "fid_worst": result.fid_worst,
                    "kid_mean": result.kid_mean,
                    "kid_mean_std": result.kid_mean_std,
                    "kid_mean_best": result.kid_mean_best,
                    "kid_mean_worst": result.kid_mean_worst,
                    "precision": result.precision,
                    "precision_std": result.precision_std,
                    "precision_best": result.precision_best,
                    "precision_worst": result.precision_worst,
                    "recall": result.recall,
                    "recall_std": result.recall_std,
                    "recall_best": result.recall_best,
                    "recall_worst": result.recall_worst,
                },
                "by_seed": result.by_seed,
            },
            f,
            indent=2,
        )

    csv_path = output_dir / "seed_metrics_latest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "seed", "fid", "kid_mean", "kid_std", "precision", "recall"])
        writer.writeheader()
        for seed, values in sorted(result.by_seed.items()):
            writer.writerow({
                "epoch": result.epoch,
                "seed": seed,
                "fid": values["fid"],
                "kid_mean": values["kid_mean"],
                "kid_std": values["kid_std"],
                "precision": values.get("precision", ""),
                "recall": values.get("recall", ""),
            })
    return json_path, csv_path
