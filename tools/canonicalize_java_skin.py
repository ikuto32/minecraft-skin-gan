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


def add_cuboid_net(mask: np.ndarray, *, x: int, y: int, w: int, d: int, h: int) -> None:
    """
    Add the used pixels of a Minecraft cuboid UV net.

    Minecraft UV net layout, excluding unused upper-left and upper-right corners:

        row 0, height d:
          [unused d] [top w] [bottom w] [unused d]

        row 1, height h:
          [side d] [front w] [side d] [back w]

    Total net size:
      width  = 2 * (w + d)
      height = d + h

    Coordinates are [x0:x1), [y0:y1).
    """
    # Top face.
    add_box(mask, x + d, y, x + d + w, y + d)

    # Bottom face.
    add_box(mask, x + d + w, y, x + d + 2 * w, y + d)

    # Side/front/back row.
    add_box(mask, x, y + d, x + 2 * (w + d), y + d + h)


def build_java_wide_arm_masks(width: int, height: int) -> RegionMasks:
    """
    Build Java Edition skin UV masks.

    Important:
      - All 64x64 skins are treated as wide-arm skins.
      - Slim-arm-specific unused pixels are intentionally ignored.
      - Head/body/arm/leg corner pixels outside the actual cuboid faces are treated as unused.
      - Coordinates are [x0:x1), [y0:y1).

    Supported sizes:
      - 64x64: modern Java skin
      - 64x32: legacy Java skin
    """
    if (width, height) not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported skin size: {width}x{height}. Expected 64x64 or 64x32.")

    base = np.zeros((height, width), dtype=bool)
    overlay = np.zeros((height, width), dtype=bool)

    # Head base and head overlay.
    # Head dimensions: w=8, d=8, h=8; net size 32x16.
    add_cuboid_net(base, x=0, y=0, w=8, d=8, h=8)
    add_cuboid_net(overlay, x=32, y=0, w=8, d=8, h=8)

    # Shared 64x32 body parts.
    # Right leg: w=4, d=4, h=12; net size 16x16.
    add_cuboid_net(base, x=0, y=16, w=4, d=4, h=12)

    # Torso: w=8, d=4, h=12; net size 24x16.
    add_cuboid_net(base, x=16, y=16, w=8, d=4, h=12)

    # Right arm, wide-arm layout: w=4, d=4, h=12; net size 16x16.
    add_cuboid_net(base, x=40, y=16, w=4, d=4, h=12)

    if height == 64:
        # 64x64 second-layer overlays and left-side limbs.
        add_cuboid_net(overlay, x=0, y=32, w=4, d=4, h=12)   # right leg overlay
        add_cuboid_net(overlay, x=16, y=32, w=8, d=4, h=12)  # torso overlay
        add_cuboid_net(overlay, x=40, y=32, w=4, d=4, h=12)  # right arm overlay

        add_cuboid_net(overlay, x=0, y=48, w=4, d=4, h=12)   # left leg overlay
        add_cuboid_net(base, x=16, y=48, w=4, d=4, h=12)     # left leg base
        add_cuboid_net(base, x=32, y=48, w=4, d=4, h=12)     # left arm base
        add_cuboid_net(overlay, x=48, y=48, w=4, d=4, h=12)  # left arm overlay

    used = base | overlay
    unused = ~used

    return RegionMasks(base=base, overlay=overlay, unused=unused)


def round_div_half_up(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    """Integer round-half-up for non-negative values."""
    return (2 * numerator + denominator) // (2 * denominator)


def quantize_inner_layer_rgb(rgb: np.ndarray, alpha_opacity: np.ndarray) -> np.ndarray:
    """
    Java Edition inner-layer color reduction.

    PNG alpha is opacity:
      - 0   = fully transparent
      - 255 = fully opaque

    For a base-layer pixel with opacity a:
      - a == 0:
          RGB becomes solid black.
      - a in 1..255:
          RGB is rounded to one of a+1 possible levels:
              round(255 * k / a), where k = round(channel * a / 255)

    Example:
      a = 2 gives levels 0, 128, 255.
      RGB(255, 13, 142) becomes RGB(255, 0, 128).
    """
    if rgb.dtype != np.uint8 or alpha_opacity.dtype != np.uint8:
        raise TypeError("rgb and alpha_opacity must be uint8 arrays")

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be shaped [N, 3], got {rgb.shape}")
    if alpha_opacity.ndim != 1 or alpha_opacity.shape[0] != rgb.shape[0]:
        raise ValueError("alpha_opacity must be shaped [N] and match rgb")

    out = np.zeros_like(rgb, dtype=np.uint8)

    visible = alpha_opacity > 0
    if not np.any(visible):
        return out

    c = rgb[visible].astype(np.uint32)
    a = alpha_opacity[visible].astype(np.uint32)[:, None]

    # k = round(channel * opacity / 255), k in [0, opacity].
    k = round_div_half_up(c * a, 255)

    # q = round(255 * k / opacity), q in [0, 255].
    q = round_div_half_up(255 * k, a)
    out[visible] = np.clip(q, 0, 255).astype(np.uint8)

    return out


def canonicalize_java_skin_rgba(arr: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """
    Canonicalize an RGBA Minecraft skin according to Java Edition constraints.

    Rules:
      1. wide arms are assumed; slim arms are not distinguished.
      2. base/inner layer:
           - use original PNG alpha as opacity for RGB quantization.
           - alpha=0 becomes solid black.
           - final alpha is 255, because Java inner layer cannot be transparent.
      3. unused region:
           - alpha is set to 0.
           - RGB is set to 0.
      4. overlay/outer layer:
           - if alpha == 0, set RGB to 0.
           - otherwise preserve RGB and alpha.
    """
    if arr.dtype != np.uint8:
        raise TypeError("Expected uint8 RGBA array")
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA array, got {arr.shape}")

    height, width = arr.shape[:2]
    masks = build_java_wide_arm_masks(width, height)

    original = arr
    out = arr.copy()

    rgb_in = original[:, :, :3]
    alpha_in = original[:, :, 3]

    rgb_out = out[:, :, :3]
    alpha_out = out[:, :, 3]

    base = masks.base
    overlay = masks.overlay
    unused = masks.unused

    base_alpha_before = alpha_in[base].copy()
    base_rgb_before = rgb_in[base].copy()

    unused_alpha_before = alpha_in[unused].copy()
    unused_rgb_before = rgb_in[unused].copy()

    overlay_alpha_before = alpha_in[overlay].copy()
    overlay_rgb_before = rgb_in[overlay].copy()

    # Base / inner layer.
    rgb_out[base] = quantize_inner_layer_rgb(rgb_in[base], alpha_in[base])
    alpha_out[base] = 255

    # Unused region: canonical transparent black.
    rgb_out[unused] = 0
    alpha_out[unused] = 0

    # Overlay / outer layer: clear hidden RGB only for fully transparent pixels.
    overlay_transparent = alpha_out[overlay] == 0
    overlay_rgb_work = rgb_out[overlay]
    overlay_rgb_work[overlay_transparent] = 0
    rgb_out[overlay] = overlay_rgb_work

    stats = {
        "changed": int(np.any(out != original)),
        "changed_pixels": int(np.count_nonzero(np.any(out != original, axis=2))),

        "base_pixels": int(np.count_nonzero(base)),
        "base_alpha_zero_pixels": int(np.count_nonzero(base_alpha_before == 0)),
        "base_alpha_semitransparent_pixels": int(np.count_nonzero((base_alpha_before > 0) & (base_alpha_before < 255))),
        "base_alpha_not_opaque_pixels": int(np.count_nonzero(base_alpha_before < 255)),
        "base_alpha_changed_pixels": int(np.count_nonzero(alpha_out[base] != base_alpha_before)),
        "base_rgb_changed_pixels": int(np.count_nonzero(np.any(rgb_out[base] != base_rgb_before, axis=1))),

        "unused_pixels": int(np.count_nonzero(unused)),
        "unused_alpha_positive_pixels": int(np.count_nonzero(unused_alpha_before > 0)),
        "unused_rgb_nonzero_pixels": int(np.count_nonzero(np.any(unused_rgb_before != 0, axis=1))),
        "unused_changed_pixels": int(np.count_nonzero(np.any(out[unused] != original[unused], axis=1))),

        "overlay_pixels": int(np.count_nonzero(overlay)),
        "overlay_alpha_zero_pixels": int(np.count_nonzero(overlay_alpha_before == 0)),
        "overlay_alpha_zero_rgb_nonzero_pixels": int(
            np.count_nonzero((overlay_alpha_before == 0) & np.any(overlay_rgb_before != 0, axis=1))
        ),
        "overlay_rgb_changed_pixels": int(np.count_nonzero(np.any(rgb_out[overlay] != overlay_rgb_before, axis=1))),
    }

    return out, stats


def iter_png_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"Input file is not PNG: {input_path}")
        return [input_path]

    pattern = "**/*.png" if recursive else "*.png"
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


def load_processed_sources(jsonl_path: Path) -> set[str]:
    processed: set[str] = set()
    if not jsonl_path.exists():
        return processed

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = rec.get("source")
            if isinstance(source, str):
                processed.add(source)
    return processed


def resolve_output_path(source: Path, input_root: Path, output_dir: Path, in_place: bool) -> Path:
    if in_place:
        return source

    if input_root.is_file():
        return output_dir / source.name

    return output_dir / source.relative_to(input_root)


def atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def process_one(
    source: Path,
    *,
    input_root: Path,
    output_dir: Path,
    in_place: bool,
    overwrite: bool,
    dry_run: bool,
    copy_unsupported: bool,
) -> dict[str, Any]:
    output_path = resolve_output_path(source, input_root, output_dir, in_place)

    rec: dict[str, Any] = {
        "source": str(source),
        "output": str(output_path),
        "status": "unknown",
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }

    if output_path.exists() and not overwrite and not in_place and not dry_run:
        rec["status"] = "skipped_output_exists"
        return rec

    try:
        with Image.open(source) as img:
            rec["mode"] = img.mode
            rec["width"], rec["height"] = img.size

            if img.size not in SUPPORTED_SIZES:
                rec["status"] = "unsupported_size"
                rec["error"] = f"unsupported size: {img.size[0]}x{img.size[1]}"
                if copy_unsupported and not dry_run and not in_place:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if overwrite or not output_path.exists():
                        output_path.write_bytes(source.read_bytes())
                return rec

            arr = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()

        fixed, stats = canonicalize_java_skin_rgba(arr)
        rec.update(stats)

        if dry_run:
            rec["status"] = "dry_run_changed" if stats["changed"] else "dry_run_unchanged"
            return rec

        atomic_save_png(Image.fromarray(fixed, mode="RGBA"), output_path)
        rec["status"] = "changed" if stats["changed"] else "unchanged"
        return rec

    except UnidentifiedImageError as exc:
        rec["status"] = "unreadable"
        rec["error"] = str(exc)
        return rec
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


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
        "base_alpha_not_opaque_pixels",
        "base_alpha_changed_pixels",
        "base_rgb_changed_pixels",
        "unused_alpha_positive_pixels",
        "unused_rgb_nonzero_pixels",
        "unused_changed_pixels",
        "overlay_alpha_zero_pixels",
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
            writer.writerow({field: rec.get(field) for field in fields})


def write_summary_from_jsonl(jsonl_path: Path, summary_path: Path) -> None:
    status_counts: dict[str, int] = {}
    sums: dict[str, int] = {}
    sum_fields = [
        "changed_pixels",
        "base_alpha_zero_pixels",
        "base_alpha_semitransparent_pixels",
        "base_alpha_not_opaque_pixels",
        "base_alpha_changed_pixels",
        "base_rgb_changed_pixels",
        "unused_alpha_positive_pixels",
        "unused_rgb_nonzero_pixels",
        "unused_changed_pixels",
        "overlay_alpha_zero_pixels",
        "overlay_alpha_zero_rgb_nonzero_pixels",
        "overlay_rgb_changed_pixels",
    ]

    total = 0
    with jsonl_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1

            status = str(rec.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

            for field in sum_fields:
                value = rec.get(field)
                if isinstance(value, int):
                    sums[field] = sums.get(field, 0) + value

    summary = {
        "total_records": total,
        "status_counts": dict(sorted(status_counts.items())),
        "sums": sums,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def self_test_quantization() -> None:
    rgb = np.array([[255, 13, 142], [10, 20, 30], [255, 255, 255]], dtype=np.uint8)
    alpha = np.array([2, 0, 255], dtype=np.uint8)
    got = quantize_inner_layer_rgb(rgb, alpha)
    expected = np.array([[255, 0, 128], [0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    if not np.array_equal(got, expected):
        raise AssertionError(f"quantization self-test failed: got={got.tolist()}, expected={expected.tolist()}")

    masks = build_java_wide_arm_masks(64, 64)

    # Head base top-row corners must be unused.
    assert masks.unused[0, 0]
    assert masks.unused[0, 7]
    assert masks.unused[0, 24]
    assert masks.unused[0, 31]

    # Head top/bottom faces must be base.
    assert masks.base[0, 8]
    assert masks.base[0, 23]

    # Body top-row corners must be unused.
    assert masks.unused[16, 16]
    assert masks.unused[16, 39]

    # Right arm top-row corners must be unused.
    assert masks.unused[16, 40]
    assert masks.unused[16, 55]

    # Left leg and left arm lower-row top corners must be unused.
    assert masks.unused[48, 16]
    assert masks.unused[48, 31]
    assert masks.unused[48, 32]
    assert masks.unused[48, 47]

    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize Minecraft Java Edition skin PNGs using wide-arm UV constraints."
    )
    parser.add_argument("--input", type=Path, required=False, help="Input PNG file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/java_skins"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/java_skin_canonicalize_logs"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input files atomically.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy-unsupported", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test_quantization()
        return

    if args.input is None:
        raise SystemExit("--input is required unless --self-test is used")

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

    processed = load_processed_sources(jsonl_path) if args.resume else set()
    to_process = [p for p in files if str(p) not in processed]

    mode = "a" if args.resume else "w"
    with jsonl_path.open(mode, encoding="utf-8") as log:
        progress = tqdm(to_process, desc="Canonicalizing Java skins", unit="file")
        for i, source in enumerate(progress, start=1):
            rec = process_one(
                source,
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
