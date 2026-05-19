from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SUPPORTED_SIZES = {(64, 64), (64, 32)}


@dataclass(frozen=True)
class RegionMasks:
    base: np.ndarray
    overlay: np.ndarray
    unused: np.ndarray


def add_box(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    mask[y0:y1, x0:x1] = True


def build_java_wide_arm_masks(width: int, height: int) -> RegionMasks:
    """
    Java Edition canonical UV masks.

    This intentionally treats every 64x64 skin as the wide-arm layout.
    Slim-arm-specific unused areas are not modeled, by design.

    Supported:
      - 64x64 modern Java skin
      - 64x32 legacy Java skin

    Coordinate convention:
      x0/y0 inclusive, x1/y1 exclusive.
    """
    if (width, height) not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported skin size: {width}x{height}. Expected 64x64 or 64x32.")

    base = np.zeros((height, width), dtype=bool)
    overlay = np.zeros((height, width), dtype=bool)
    unused = np.zeros((height, width), dtype=bool)

    # Head base and head overlay.
    add_box(base, 0, 0, 32, 16)
    add_box(overlay, 32, 0, 64, 16)

    # Legacy-compatible base body parts.
    add_box(base, 0, 16, 16, 32)    # right leg base
    add_box(base, 16, 16, 40, 32)   # torso base
    add_box(base, 40, 16, 56, 32)   # right arm base, wide-arm layout

    if height == 32:
        add_box(unused, 56, 16, 64, 32)
    else:
        # 64x64 modern layout.
        add_box(overlay, 0, 32, 16, 48)   # right leg overlay
        add_box(overlay, 16, 32, 40, 48)  # torso overlay
        add_box(overlay, 40, 32, 56, 48)  # right arm overlay, wide-arm layout

        add_box(overlay, 0, 48, 16, 64)   # left leg overlay
        add_box(base, 16, 48, 32, 64)     # left leg base
        add_box(base, 32, 48, 48, 64)     # left arm base, wide-arm layout
        add_box(overlay, 48, 48, 64, 64)  # left arm overlay, wide-arm layout

        add_box(unused, 56, 16, 64, 48)

    return RegionMasks(base=base, overlay=overlay, unused=unused)


def quantize_inner_layer_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply Java Edition inner-layer opacity quantization.

    For each base-layer pixel:
      - alpha == 0:
          RGB becomes solid black.
      - alpha in 1..254:
          channel c is rounded to one of:
              round(255 * k / alpha), k in [0, alpha]
          where k is round(c * alpha / 255).
          Output alpha is handled outside this function.
      - alpha == 255:
          RGB remains unchanged.

    This implements half-up integer rounding rather than NumPy's bankers rounding.
    """
    if rgb.dtype != np.uint8 or alpha.dtype != np.uint8:
        raise TypeError("rgb and alpha must be uint8 arrays")

    out = np.zeros_like(rgb, dtype=np.uint8)

    nonzero = alpha > 0
    if not np.any(nonzero):
        return out

    c = rgb[nonzero].astype(np.uint32)            # [N, 3]
    a = alpha[nonzero].astype(np.uint32)[:, None] # [N, 1]

    # k = round_half_up(c * a / 255)
    k = (2 * c * a + 255) // (2 * 255)

    # q = round_half_up(255 * k / a)
    q = (2 * 255 * k + a) // (2 * a)
    q = np.clip(q, 0, 255).astype(np.uint8)

    out[nonzero] = q
    return out


def canonicalize_java_skin(arr: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """
    Project an RGBA Minecraft skin image onto the requested Java Edition constraints.

    Rules:
      1. Use wide-arm Java UV masks. Slim-arm-specific unused areas are ignored.
      2. Base/inner layer:
           - RGB is quantized according to the original alpha value.
           - alpha=0 becomes solid black.
           - final alpha is 255 for all base pixels.
      3. Unused region:
           - alpha is set to 0.
           - RGB is set to 0.
      4. Overlay/outer layer:
           - if alpha == 0, RGB is set to 0.
           - otherwise RGB/alpha are preserved.
    """
    if arr.dtype != np.uint8:
        raise TypeError("Expected uint8 RGBA image array")
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA array, got {arr.shape}")

    height, width = arr.shape[:2]
    masks = build_java_wide_arm_masks(width, height)

    original = arr
    out = arr.copy()

    base = masks.base
    overlay = masks.overlay
    unused = masks.unused

    orig_alpha = original[:, :, 3]
    orig_rgb = original[:, :, :3]

    # Base layer: Java inner-layer behavior.
    base_rgb_before = out[:, :, :3][base].copy()
    base_alpha_before = out[:, :, 3][base].copy()

    quantized_base_rgb = quantize_inner_layer_rgb(orig_rgb[base], orig_alpha[base])
    out[:, :, :3][base] = quantized_base_rgb
    out[:, :, 3][base] = 255

    base_rgb_after = out[:, :, :3][base]
    base_alpha_after = out[:, :, 3][base]

    # Unused region: canonical transparent black.
    unused_rgb_nonzero_before = np.any(out[:, :, :3][unused] != 0, axis=1)
    unused_alpha_positive_before = out[:, :, 3][unused] > 0

    out[:, :, :3][unused] = 0
    out[:, :, 3][unused] = 0

    # Overlay: transparent pixels should carry no hidden RGB.
    overlay_alpha_zero = out[:, :, 3][overlay] == 0
    overlay_rgb_before = out[:, :, :3][overlay].copy()

    overlay_rgb = out[:, :, :3][overlay]
    overlay_rgb[overlay_alpha_zero] = 0
    out[:, :, :3][overlay] = overlay_rgb

    overlay_rgb_after = out[:, :, :3][overlay]

    stats = {
        "changed": int(np.any(out != original)),
        "changed_pixels": int(np.count_nonzero(np.any(out != original, axis=2))),
        "base_pixels": int(np.count_nonzero(base)),
        "base_alpha_zero_pixels": int(np.count_nonzero(base_alpha_before == 0)),
        "base_alpha_semitransparent_pixels": int(np.count_nonzero((base_alpha_before > 0) & (base_alpha_before < 255))),
        "base_alpha_changed_pixels": int(np.count_nonzero(base_alpha_before != base_alpha_after)),
        "base_rgb_changed_pixels": int(np.count_nonzero(np.any(base_rgb_before != base_rgb_after, axis=1))),
        "unused_pixels": int(np.count_nonzero(unused)),
        "unused_alpha_positive_pixels": int(np.count_nonzero(unused_alpha_positive_before)),
        "unused_rgb_nonzero_pixels": int(np.count_nonzero(unused_rgb_nonzero_before)),
        "overlay_pixels": int(np.count_nonzero(overlay)),
        "overlay_alpha_zero_rgb_nonzero_pixels": int(
            np.count_nonzero(overlay_alpha_zero & np.any(overlay_rgb_before != 0, axis=1))
        ),
        "overlay_rgb_changed_pixels": int(np.count_nonzero(np.any(overlay_rgb_before != overlay_rgb_after, axis=1))),
    }

    return out, stats


def iter_png_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"Input file is not a PNG: {input_path}")
        return [input_path]

    pattern = "**/*.png" if recursive else "*.png"
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


def load_processed_paths(log_path: Path) -> set[str]:
    processed: set[str] = set()
    if not log_path.exists():
        return processed

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = rec.get("source")
            if isinstance(source, str):
                processed.add(source)
    return processed


def atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def resolve_output_path(path: Path, input_root: Path, output_dir: Path, in_place: bool) -> Path:
    if in_place:
        return path

    if input_root.is_file():
        return output_dir / path.name

    return output_dir / path.relative_to(input_root)


def process_one(
    path: Path,
    *,
    input_root: Path,
    output_dir: Path,
    in_place: bool,
    overwrite: bool,
    dry_run: bool,
    copy_unsupported: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": str(path),
        "output": None,
        "status": "unknown",
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }

    output_path = resolve_output_path(path, input_root, output_dir, in_place)
    record["output"] = str(output_path)

    if output_path.exists() and not overwrite and not in_place and not dry_run:
        record["status"] = "skipped_output_exists"
        return record

    try:
        with Image.open(path) as img:
            record["mode"] = img.mode
            record["width"], record["height"] = img.size

            if img.size not in SUPPORTED_SIZES:
                record["status"] = "unsupported_size"
                record["error"] = f"unsupported size {img.size[0]}x{img.size[1]}"
                if copy_unsupported and not dry_run and not in_place:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if overwrite or not output_path.exists():
                        import shutil
                        shutil.copy2(path, output_path)
                return record

            rgba = img.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.uint8).copy()

        fixed, stats = canonicalize_java_skin(arr)
        record.update(stats)

        if dry_run:
            record["status"] = "dry_run_changed" if stats["changed"] else "dry_run_unchanged"
            return record

        fixed_img = Image.fromarray(fixed, mode="RGBA")
        atomic_save_png(fixed_img, output_path)

        record["status"] = "changed" if stats["changed"] else "unchanged"
        return record

    except UnidentifiedImageError as exc:
        record["status"] = "unreadable"
        record["error"] = str(exc)
        return record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    fields = [
        "source",
        "output",
        "status",
        "width",
        "height",
        "mode",
        "changed",
        "changed_pixels",
        "base_alpha_zero_pixels",
        "base_alpha_semitransparent_pixels",
        "base_alpha_changed_pixels",
        "base_rgb_changed_pixels",
        "unused_alpha_positive_pixels",
        "unused_rgb_nonzero_pixels",
        "overlay_alpha_zero_rgb_nonzero_pixels",
        "overlay_rgb_changed_pixels",
        "error",
    ]

    with jsonl_path.open("r", encoding="utf-8") as src, csv_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()

        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            writer.writerow({key: rec.get(key) for key in fields})


def write_summary_from_jsonl(jsonl_path: Path, summary_path: Path) -> None:
    counters: dict[str, int] = {}
    numeric_sums: dict[str, int] = {}

    sum_keys = [
        "changed_pixels",
        "base_alpha_zero_pixels",
        "base_alpha_semitransparent_pixels",
        "base_alpha_changed_pixels",
        "base_rgb_changed_pixels",
        "unused_alpha_positive_pixels",
        "unused_rgb_nonzero_pixels",
        "overlay_alpha_zero_rgb_nonzero_pixels",
        "overlay_rgb_changed_pixels",
    ]

    total = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            status = str(rec.get("status", "unknown"))
            counters[status] = counters.get(status, 0) + 1
            for key in sum_keys:
                value = rec.get(key)
                if isinstance(value, int):
                    numeric_sums[key] = numeric_sums.get(key, 0) + value

    summary = {
        "total_records": total,
        "status_counts": dict(sorted(counters.items())),
        "sums": numeric_sums,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize Minecraft Java Edition skin PNGs using wide-arm UV constraints."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input PNG file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/java_skins"), help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Process PNG files recursively when input is a directory.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input files atomically. Use with care.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write images; only write logs.")
    parser.add_argument("--resume", action="store_true", help="Skip files already present in the JSONL log.")
    parser.add_argument("--copy-unsupported", action="store_true", help="Copy unsupported-size PNGs unchanged to output-dir.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/java_skin_canonicalize_logs"))
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Flush logs every N files.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")

    if args.in_place and args.dry_run:
        raise ValueError("--in-place and --dry-run cannot be used together")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.log_dir / "canonicalize_results.jsonl"
    csv_path = args.log_dir / "canonicalize_results.csv"
    summary_path = args.log_dir / "summary.json"

    files = iter_png_files(input_path, args.recursive)
    if not files:
        print(f"No PNG files found: {input_path}", file=sys.stderr)
        sys.exit(1)

    processed = load_processed_paths(jsonl_path) if args.resume else set()
    to_process = [p for p in files if str(p) not in processed]

    mode = "a" if args.resume else "w"
    with jsonl_path.open(mode, encoding="utf-8") as log:
        progress = tqdm(to_process, desc="Canonicalizing Java skins", unit="file")

        for i, path in enumerate(progress, start=1):
            rec = process_one(
                path,
                input_root=input_path,
                output_dir=args.output_dir,
                in_place=args.in_place,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                copy_unsupported=args.copy_unsupported,
            )
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i % max(1, args.checkpoint_every) == 0:
                log.flush()
                os.fsync(log.fileno())

    write_csv_from_jsonl(jsonl_path, csv_path)
    write_summary_from_jsonl(jsonl_path, summary_path)

    print("Done.")
    print(f"JSONL log: {jsonl_path}")
    print(f"CSV log:   {csv_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
