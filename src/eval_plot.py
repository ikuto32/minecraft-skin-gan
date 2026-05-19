from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_empty_history(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, title in zip(axes, ["FID", "KID mean"], strict=True):
        axis.set_title(title)
        axis.text(0.5, 0.5, "No history data", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Lower is better")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metrics(history_csv: Path, out_path: Path) -> None:
    with history_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        _plot_empty_history(out_path)
        return

    epochs = [int(r["epoch"]) for r in rows]
    fids = [float(r["fid"]) for r in rows]
    fid_stds = [float(r.get("fid_std", 0.0) or 0.0) for r in rows]
    kid_means = [float(r["kid_mean"]) for r in rows]
    kid_mean_stds = [float(r.get("kid_mean_std", 0.0) or 0.0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(rows) == 1:
        axes[0].scatter(epochs, fids, marker="o")
        axes[1].scatter(epochs, kid_means, marker="o", color="tab:orange")
    else:
        axes[0].plot(epochs, fids, marker="o")
        axes[1].plot(epochs, kid_means, marker="o", color="tab:orange")
    if any(v > 0.0 for v in fid_stds):
        axes[0].fill_between(
            epochs,
            [m - s for m, s in zip(fids, fid_stds, strict=True)],
            [m + s for m, s in zip(fids, fid_stds, strict=True)],
            alpha=0.2,
        )
    if any(v > 0.0 for v in kid_mean_stds):
        axes[1].fill_between(
            epochs,
            [m - s for m, s in zip(kid_means, kid_mean_stds, strict=True)],
            [m + s for m, s in zip(kid_means, kid_mean_stds, strict=True)],
            alpha=0.2,
            color="tab:orange",
        )

    axes[0].set_title("FID")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("KID mean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Lower is better")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
