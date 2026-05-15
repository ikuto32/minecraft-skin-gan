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
    seeds: list[int] | None = None


def save_latest_metrics(result: EvalResult, output_dir: Path) -> Path:
    json_path = output_dir / "latest_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    return json_path


def append_metrics_history(result: EvalResult, output_dir: Path) -> Path:
    history_path = output_dir / "metrics_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        if write_header:
            writer.writeheader()
        row = asdict(result)
        row["seeds"] = ",".join(str(s) for s in (result.seeds or [result.seed]))
        writer.writerow(row)
    return history_path
