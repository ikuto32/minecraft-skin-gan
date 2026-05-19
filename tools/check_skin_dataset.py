from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def parse_size_spec(spec: str) -> set[tuple[int, int]]:
    sizes: set[tuple[int, int]] = set()
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid size spec: {item!r}. Use like 64x64,64x32")
        w, h = item.split("x", 1)
        sizes.add((int(w), int(h)))
    if not sizes:
        raise ValueError("At least one allowed size is required.")
    return sizes


def find_png_files(data_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    else:
        files = [p for p in data_dir.glob("*") if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(files)


def add_box(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    mask[y0:y1, x0:x1] = True


def minecraft_layout_masks(width: int, height: int) -> dict[str, np.ndarray] | None:
    """
    Canonical Minecraft Java skin masks.

    64x64 layout:
      base:
        head, torso, right arm, right leg, left arm, left leg
      overlay:
        hat/head overlay, jacket/body overlay, arm/leg overlays
      unused:
        x=56..63, y=16..47

    64x32 legacy layout:
      base:
        head, torso, right arm, right leg
      overlay:
        hat/head overlay only
      unused:
        x=56..63, y=16..31
    """
    if (width, height) not in {(64, 64), (64, 32)}:
        return None

    base = np.zeros((height, width), dtype=bool)
    overlay = np.zeros((height, width), dtype=bool)
    unused = np.zeros((height, width), dtype=bool)

    # Head base and head overlay.
    add_box(base, 0, 0, 32, 16)
    add_box(overlay, 32, 0, 64, 16)

    # Legacy-compatible base body parts.
    add_box(base, 0, 16, 16, 32)    # right leg base
    add_box(base, 16, 16, 40, 32)   # torso base
    add_box(base, 40, 16, 56, 32)   # right arm base

    if height == 32:
        add_box(unused, 56, 16, 64, 32)
    else:
        # 64x64 second layer / left limbs.
        add_box(overlay, 0, 32, 16, 48)   # right leg overlay
        add_box(overlay, 16, 32, 40, 48)  # torso overlay
        add_box(overlay, 40, 32, 56, 48)  # right arm overlay

        add_box(overlay, 0, 48, 16, 64)   # left leg overlay
        add_box(base, 16, 48, 32, 64)     # left leg base
        add_box(base, 32, 48, 48, 64)     # left arm base
        add_box(overlay, 48, 48, 64, 64)  # left arm overlay

        add_box(unused, 56, 16, 64, 48)

    used = base | overlay
    other = ~(used | unused)

    return {
        "base": base,
        "overlay": overlay,
        "unused": unused,
        "used": used,
        "other": other,
    }


def normalized_rgba_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def issue(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def check_one_file(
    path: Path,
    *,
    allowed_sizes: set[tuple[int, int]],
    base_alpha_min: int,
    unused_alpha_max: int,
    require_binary_alpha: bool,
    require_transparent_rgb_zero: bool,
    seen_hashes: dict[str, str],
    check_duplicates: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    issues: list[dict[str, str]] = []

    record: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "width": None,
        "height": None,
        "mode": None,
        "file_sha256": None,
        "content_sha256": None,
        "duplicate_of": None,
        "metrics": {},
        "issues": issues,
    }

    try:
        record["file_sha256"] = file_sha256(path)
        with Image.open(path) as img:
            record["mode"] = img.mode
            width, height = img.size
            record["width"] = width
            record["height"] = height

            if img.mode != "RGBA":
                issues.append(issue("warning", "mode_not_rgba", f"image mode is {img.mode}, expected RGBA"))

            rgba = img.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.uint8)

    except UnidentifiedImageError as exc:
        issues.append(issue("error", "unreadable_png", f"PIL cannot identify image: {exc}"))
        return finalize_record(record), None
    except Exception as exc:
        issues.append(issue("error", "read_error", f"failed to read image: {type(exc).__name__}: {exc}"))
        return finalize_record(record), None

    width = int(record["width"])
    height = int(record["height"])

    if (width, height) not in allowed_sizes:
        allowed_text = ",".join(f"{w}x{h}" for w, h in sorted(allowed_sizes))
        issues.append(issue("error", "invalid_size", f"size is {width}x{height}, allowed sizes are {allowed_text}"))

    record["content_sha256"] = normalized_rgba_hash(arr)

    if check_duplicates:
        h = record["content_sha256"]
        if h in seen_hashes:
            record["duplicate_of"] = seen_hashes[h]
            issues.append(issue("warning", "duplicate_rgba", f"duplicate normalized RGBA content of {seen_hashes[h]}"))
        else:
            seen_hashes[h] = str(path)

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    unique_alpha = np.unique(alpha)
    record["metrics"]["alpha_unique_count"] = int(unique_alpha.size)
    record["metrics"]["alpha_min"] = int(alpha.min())
    record["metrics"]["alpha_max"] = int(alpha.max())
    record["metrics"]["transparent_pixels"] = int(np.count_nonzero(alpha == 0))
    record["metrics"]["opaque_pixels"] = int(np.count_nonzero(alpha == 255))

    if require_binary_alpha:
        non_binary_alpha = int(np.count_nonzero((alpha != 0) & (alpha != 255)))
        record["metrics"]["non_binary_alpha_pixels"] = non_binary_alpha
        if non_binary_alpha > 0:
            issues.append(issue("warning", "non_binary_alpha", f"{non_binary_alpha} pixels have alpha other than 0 or 255"))

    if require_transparent_rgb_zero:
        transparent = alpha == 0
        transparent_rgb_nonzero = int(np.count_nonzero(transparent & np.any(rgb != 0, axis=2)))
        record["metrics"]["transparent_rgb_nonzero_pixels"] = transparent_rgb_nonzero
        if transparent_rgb_nonzero > 0:
            issues.append(
                issue(
                    "warning",
                    "transparent_rgb_nonzero",
                    f"{transparent_rgb_nonzero} transparent pixels have non-zero RGB",
                )
            )

    masks = minecraft_layout_masks(width, height)
    if masks is None:
        issues.append(issue("warning", "unknown_layout", f"no built-in UV layout mask for {width}x{height}"))
    else:
        base = masks["base"]
        overlay = masks["overlay"]
        unused = masks["unused"]

        base_bad_alpha = int(np.count_nonzero(alpha[base] < base_alpha_min))
        unused_bad_alpha = int(np.count_nonzero(alpha[unused] > unused_alpha_max))
        overlay_visible = int(np.count_nonzero(alpha[overlay] > 0))
        overlay_total = int(np.count_nonzero(overlay))
        unused_total = int(np.count_nonzero(unused))

        record["metrics"]["base_alpha_below_min_pixels"] = base_bad_alpha
        record["metrics"]["unused_alpha_above_max_pixels"] = unused_bad_alpha
        record["metrics"]["overlay_visible_pixels"] = overlay_visible
        record["metrics"]["overlay_visible_ratio"] = float(overlay_visible / max(1, overlay_total))
        record["metrics"]["unused_pixels"] = unused_total

        if base_bad_alpha > 0:
            issues.append(
                issue(
                    "error",
                    "base_layer_transparent",
                    f"{base_bad_alpha} base-layer pixels have alpha < {base_alpha_min}",
                )
            )

        if unused_bad_alpha > 0:
            issues.append(
                issue(
                    "error",
                    "unused_region_not_transparent",
                    f"{unused_bad_alpha} unused-region pixels have alpha > {unused_alpha_max}",
                )
            )

        if require_transparent_rgb_zero:
            unused_rgb_nonzero = int(np.count_nonzero(unused & np.any(rgb != 0, axis=2)))
            record["metrics"]["unused_rgb_nonzero_pixels"] = unused_rgb_nonzero
            if unused_rgb_nonzero > 0:
                issues.append(
                    issue(
                        "warning",
                        "unused_rgb_nonzero",
                        f"{unused_rgb_nonzero} unused-region pixels have non-zero RGB",
                    )
                )

    return finalize_record(record), arr


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    issues = record["issues"]
    n_errors = sum(1 for x in issues if x["level"] == "error")
    n_warnings = sum(1 for x in issues if x["level"] == "warning")
    record["n_errors"] = n_errors
    record["n_warnings"] = n_warnings
    record["ok"] = n_errors == 0
    return record


def init_pixel_stats(width: int, height: int) -> dict[str, Any]:
    return {
        "width": width,
        "height": height,
        "n_images": 0,
        "alpha_nonzero": np.zeros((height, width), dtype=np.uint64),
        "alpha_opaque": np.zeros((height, width), dtype=np.uint64),
        "rgb_sum": np.zeros((height, width, 3), dtype=np.float64),
        "rgb_sumsq": np.zeros((height, width, 3), dtype=np.float64),
    }


def update_pixel_stats(stats: dict[str, Any], arr: np.ndarray) -> None:
    height, width = arr.shape[:2]
    if width != stats["width"] or height != stats["height"]:
        return

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.float64)

    stats["n_images"] += 1
    stats["alpha_nonzero"] += (alpha > 0).astype(np.uint64)
    stats["alpha_opaque"] += (alpha == 255).astype(np.uint64)
    stats["rgb_sum"] += rgb
    stats["rgb_sumsq"] += rgb * rgb


def save_pixel_stats_npz(path: Path, stats: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        width=np.array(stats["width"], dtype=np.int64),
        height=np.array(stats["height"], dtype=np.int64),
        n_images=np.array(stats["n_images"], dtype=np.int64),
        alpha_nonzero=stats["alpha_nonzero"],
        alpha_opaque=stats["alpha_opaque"],
        rgb_sum=stats["rgb_sum"],
        rgb_sumsq=stats["rgb_sumsq"],
    )


def load_pixel_stats_npz(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "n_images": int(data["n_images"]),
        "alpha_nonzero": data["alpha_nonzero"].astype(np.uint64),
        "alpha_opaque": data["alpha_opaque"].astype(np.uint64),
        "rgb_sum": data["rgb_sum"].astype(np.float64),
        "rgb_sumsq": data["rgb_sumsq"].astype(np.float64),
    }


def truncate_jsonl_to_lines(path: Path, keep_lines: int) -> None:
    if not path.exists():
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= keep_lines:
                break
            dst.write(line)
    tmp.replace(path)


def load_seen_hashes_from_jsonl(path: Path) -> dict[str, str]:
    seen: dict[str, str] = {}
    if not path.exists():
        return seen

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            h = rec.get("content_sha256")
            p = rec.get("path")
            if h and p and h not in seen:
                seen[h] = p
    return seen


def save_checkpoint(path: Path, next_index: int) -> None:
    payload = {"next_index": next_index}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["next_index"])


def write_reports(
    *,
    jsonl_path: Path,
    csv_path: Path,
    summary_path: Path,
    pixel_stats_path: Path | None,
    alpha_occupancy_csv: Path | None,
    fixed_rgb_std_threshold: float,
) -> None:
    total = 0
    ok = 0
    errors = 0
    warnings = 0
    issue_counts: Counter[str] = Counter()

    csv_columns = [
        "path",
        "ok",
        "n_errors",
        "n_warnings",
        "width",
        "height",
        "mode",
        "duplicate_of",
        "issues",
        "base_alpha_below_min_pixels",
        "unused_alpha_above_max_pixels",
        "transparent_rgb_nonzero_pixels",
        "unused_rgb_nonzero_pixels",
        "non_binary_alpha_pixels",
        "overlay_visible_ratio",
        "content_sha256",
        "file_sha256",
    ]

    with jsonl_path.open("r", encoding="utf-8") as src, csv_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=csv_columns)
        writer.writeheader()

        for line in src:
            if not line.strip():
                continue

            rec = json.loads(line)
            total += 1
            ok += int(bool(rec.get("ok")))
            errors += int(rec.get("n_errors", 0))
            warnings += int(rec.get("n_warnings", 0))

            for item in rec.get("issues", []):
                issue_counts[item.get("code", "unknown")] += 1

            metrics = rec.get("metrics", {})
            row = {
                "path": rec.get("path"),
                "ok": rec.get("ok"),
                "n_errors": rec.get("n_errors"),
                "n_warnings": rec.get("n_warnings"),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "mode": rec.get("mode"),
                "duplicate_of": rec.get("duplicate_of"),
                "issues": " | ".join(f"{x['level']}:{x['code']}:{x['message']}" for x in rec.get("issues", [])),
                "base_alpha_below_min_pixels": metrics.get("base_alpha_below_min_pixels"),
                "unused_alpha_above_max_pixels": metrics.get("unused_alpha_above_max_pixels"),
                "transparent_rgb_nonzero_pixels": metrics.get("transparent_rgb_nonzero_pixels"),
                "unused_rgb_nonzero_pixels": metrics.get("unused_rgb_nonzero_pixels"),
                "non_binary_alpha_pixels": metrics.get("non_binary_alpha_pixels"),
                "overlay_visible_ratio": metrics.get("overlay_visible_ratio"),
                "content_sha256": rec.get("content_sha256"),
                "file_sha256": rec.get("file_sha256"),
            }
            writer.writerow(row)

    summary: dict[str, Any] = {
        "total_files": total,
        "ok_files": ok,
        "files_with_errors": total - ok,
        "total_error_issues": errors,
        "total_warning_issues": warnings,
        "issue_counts": dict(issue_counts.most_common()),
    }

    if pixel_stats_path is not None and pixel_stats_path.exists():
        stats = load_pixel_stats_npz(pixel_stats_path)
        n = int(stats["n_images"])
        if n > 0:
            alpha_nonzero_rate = stats["alpha_nonzero"].astype(np.float64) / n
            alpha_opaque_rate = stats["alpha_opaque"].astype(np.float64) / n

            rgb_mean = stats["rgb_sum"] / n
            rgb_var = np.maximum(stats["rgb_sumsq"] / n - rgb_mean * rgb_mean, 0.0)
            rgb_std_mean = np.sqrt(rgb_var).mean(axis=2)

            always_transparent = alpha_nonzero_rate == 0.0
            always_opaque = alpha_opaque_rate == 1.0
            nearly_fixed_rgb = rgb_std_mean <= fixed_rgb_std_threshold

            summary["pixel_stats"] = {
                "n_images": n,
                "always_transparent_pixels": int(always_transparent.sum()),
                "always_opaque_pixels": int(always_opaque.sum()),
                "nearly_fixed_rgb_pixels": int(nearly_fixed_rgb.sum()),
                "fixed_rgb_std_threshold": fixed_rgb_std_threshold,
            }

            if alpha_occupancy_csv is not None:
                masks = minecraft_layout_masks(int(stats["width"]), int(stats["height"]))
                with alpha_occupancy_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "x",
                            "y",
                            "region",
                            "alpha_nonzero_rate",
                            "alpha_opaque_rate",
                            "rgb_std_mean",
                            "nearly_fixed_rgb",
                        ],
                    )
                    writer.writeheader()

                    height = int(stats["height"])
                    width = int(stats["width"])
                    for y in range(height):
                        for x in range(width):
                            region = "unknown"
                            if masks is not None:
                                if masks["base"][y, x]:
                                    region = "base"
                                elif masks["overlay"][y, x]:
                                    region = "overlay"
                                elif masks["unused"][y, x]:
                                    region = "unused"
                                elif masks["other"][y, x]:
                                    region = "other"

                            writer.writerow(
                                {
                                    "x": x,
                                    "y": y,
                                    "region": region,
                                    "alpha_nonzero_rate": float(alpha_nonzero_rate[y, x]),
                                    "alpha_opaque_rate": float(alpha_opaque_rate[y, x]),
                                    "rgb_std_mean": float(rgb_std_mean[y, x]),
                                    "nearly_fixed_rgb": bool(nearly_fixed_rgb[y, x]),
                                }
                            )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Minecraft skin dataset constraints.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/skin_constraint_report"))
    parser.add_argument("--allowed-sizes", type=str, default="64x64")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--base-alpha-min", type=int, default=250)
    parser.add_argument("--unused-alpha-max", type=int, default=0)
    parser.add_argument("--allow-non-binary-alpha", action="store_true")
    parser.add_argument("--allow-transparent-rgb", action="store_true")
    parser.add_argument("--no-duplicate-check", action="store_true")

    parser.add_argument("--pixel-stats-size", type=str, default="64x64")
    parser.add_argument("--no-pixel-stats", action="store_true")
    parser.add_argument("--fixed-rgb-std-threshold", type=float, default=1.0)

    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=0)

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {args.data_dir}")

    allowed_sizes = parse_size_spec(args.allowed_sizes)
    pixel_stats_size = next(iter(parse_size_spec(args.pixel_stats_size)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "per_file_results.jsonl"
    csv_path = args.output_dir / "per_file_results.csv"
    summary_path = args.output_dir / "summary.json"
    checkpoint_path = args.output_dir / "checkpoint.json"
    pixel_stats_path = args.output_dir / "pixel_stats_checkpoint.npz"
    alpha_occupancy_csv = args.output_dir / "pixel_occupancy.csv"

    files = find_png_files(args.data_dir, recursive=args.recursive)
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise RuntimeError(f"No PNG files found under {args.data_dir}")

    start_index = 0
    if args.resume and checkpoint_path.exists():
        start_index = load_checkpoint(checkpoint_path)
        truncate_jsonl_to_lines(jsonl_path, start_index)
    elif not args.resume:
        for p in [jsonl_path, csv_path, summary_path, checkpoint_path, pixel_stats_path, alpha_occupancy_csv]:
            if p.exists():
                p.unlink()

    seen_hashes = load_seen_hashes_from_jsonl(jsonl_path)

    pixel_stats: dict[str, Any] | None = None
    if not args.no_pixel_stats:
        if args.resume and pixel_stats_path.exists():
            pixel_stats = load_pixel_stats_npz(pixel_stats_path)
        else:
            w, h = pixel_stats_size
            pixel_stats = init_pixel_stats(w, h)

    mode = "a" if args.resume and jsonl_path.exists() else "w"
    with jsonl_path.open(mode, encoding="utf-8") as out:
        progress = tqdm(
            enumerate(files[start_index:], start=start_index),
            total=max(0, len(files) - start_index),
            desc="Checking skins",
            unit="file",
        )

        for index, path in progress:
            record, arr = check_one_file(
                path,
                allowed_sizes=allowed_sizes,
                base_alpha_min=args.base_alpha_min,
                unused_alpha_max=args.unused_alpha_max,
                require_binary_alpha=not args.allow_non_binary_alpha,
                require_transparent_rgb_zero=not args.allow_transparent_rgb,
                seen_hashes=seen_hashes,
                check_duplicates=not args.no_duplicate_check,
            )

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if pixel_stats is not None and arr is not None:
                update_pixel_stats(pixel_stats, arr)

            next_index = index + 1
            if args.checkpoint_every > 0 and next_index % args.checkpoint_every == 0:
                out.flush()
                os.fsync(out.fileno())
                save_checkpoint(checkpoint_path, next_index)
                if pixel_stats is not None:
                    save_pixel_stats_npz(pixel_stats_path, pixel_stats)

    save_checkpoint(checkpoint_path, len(files))
    if pixel_stats is not None:
        save_pixel_stats_npz(pixel_stats_path, pixel_stats)

    write_reports(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        summary_path=summary_path,
        pixel_stats_path=None if args.no_pixel_stats else pixel_stats_path,
        alpha_occupancy_csv=None if args.no_pixel_stats else alpha_occupancy_csv,
        fixed_rgb_std_threshold=args.fixed_rgb_std_threshold,
    )

    print(f"Done.")
    print(f"Per-file JSONL: {jsonl_path}")
    print(f"Per-file CSV:   {csv_path}")
    print(f"Summary JSON:   {summary_path}")
    if not args.no_pixel_stats:
        print(f"Pixel CSV:      {alpha_occupancy_csv}")


if __name__ == "__main__":
    main()