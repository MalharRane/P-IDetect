"""One-command Phase 0 dataset build: download -> tile -> synthetic -> merge.

Runs the full pipeline with a fixed seed so every team member and Colab
session gets bit-identical splits. Skips steps whose outputs already exist;
use --force to rebuild from scratch.

Usage (from project root):
    python scripts/build_dataset.py
    python scripts/build_dataset.py --force
    python scripts/build_dataset.py --synth-n 400 --seed 0
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

# Project root importable when run as a plain script (not module)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pidetect.data.tiling  import slice_dataset
from src.pidetect.data.synth   import build_glyph_library, generate_dataset
from src.pidetect.data.inspect import save_class_distribution

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SRC          = Path("data/digitize-pid-yolo/DigitizePID_Dataset")
TILED        = Path("data/tiled")
SYNTH        = Path("data/synthetic")
SYNTH_TILED  = Path("data/synthetic_tiled")   # 640px tiles of the synthetic sheets
MERGED       = Path("data/merged")
YAML         = Path("configs/yolo_baseline.yaml")
NC           = 32
SEP          = "=" * 62


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_files(d: Path, glob: str = "*.jpg") -> bool:
    return d.is_dir() and any(d.glob(glob))


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src -> dst; fall back to copy if cross-device."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def step_download(force: bool) -> None:
    print(f"\n{SEP}")
    print("Step 1: Download HF dataset  (hamzas/digitize-pid-yolo)")
    print(SEP)
    dl_dest = Path("data/digitize-pid-yolo")
    if not force and dl_dest.exists():
        print(f"  SKIP: {dl_dest} already exists")
        return
    from huggingface_hub import snapshot_download
    print(f"  Downloading -> {dl_dest} ...")
    snapshot_download(repo_id="hamzas/digitize-pid-yolo", repo_type="dataset",
                      local_dir=str(dl_dest))
    print("  Done.")


# ---------------------------------------------------------------------------
# Step 2 — Tile real dataset
# ---------------------------------------------------------------------------

def step_tile(force: bool) -> None:
    print(f"\n{SEP}")
    print("Step 2: Tile real dataset (640px / 20% overlap)")
    print(SEP)
    if not force and _has_files(TILED / "images/train"):
        print(f"  SKIP: {TILED}/images/train already has tiles")
        return
    if force:
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = TILED / sub / split
                if d.exists():
                    shutil.rmtree(d)

    print(f"  Source: {SRC}")
    results = slice_dataset(SRC / "images", SRC / "labels", TILED)
    for split, s in results.items():
        ret = s["boxes_out"] / max(s["boxes_in"], 1)
        print(f"  [{split}]  {s['tiles_written']:>6} tiles  "
              f"{s['boxes_out']:>7} boxes  (retention {ret:.3f}x)")


# ---------------------------------------------------------------------------
# Step 3 — Carve test split from tiled val (at source-image granularity)
# ---------------------------------------------------------------------------

def step_carve_test(seed: int, test_fraction: float, force: bool) -> None:
    print(f"\n{SEP}")
    print("Step 3: Carve test split from tiled val  (real data only)")
    print(SEP)
    if not force and _has_files(TILED / "images/test"):
        print(f"  SKIP: {TILED}/images/test already has tiles")
        return

    val_tiles = sorted((TILED / "images/val").glob("*.jpg"))
    if not val_tiles:
        raise RuntimeError(
            f"No tiles in {TILED}/images/val -- step 2 must run first."
        )

    # Tile filenames: "{original_stem}_{row:05d}.jpg"
    # rsplit("_", 1) cleanly separates stem from row index
    src_stems = sorted({p.stem.rsplit("_", 1)[0] for p in val_tiles})
    rng_r = random.Random(seed)
    rng_r.shuffle(src_stems)
    n_test   = max(1, int(len(src_stems) * test_fraction))
    test_set = set(src_stems[:n_test])

    print(f"  val source images : {len(src_stems)}")
    print(f"  -> test sources   : {n_test}  ({test_fraction:.0%})")
    print(f"  -> val sources    : {len(src_stems) - n_test}")

    (TILED / "images/test").mkdir(parents=True, exist_ok=True)
    (TILED / "labels/test").mkdir(parents=True, exist_ok=True)

    moved = 0
    for tile in val_tiles:
        if tile.stem.rsplit("_", 1)[0] in test_set:
            tile.rename(TILED / "images/test" / tile.name)
            lbl = TILED / "labels/val" / (tile.stem + ".txt")
            if lbl.exists():
                lbl.rename(TILED / "labels/test" / lbl.name)
            moved += 1

    remaining = len(list((TILED / "images/val").glob("*.jpg")))
    print(f"  tiles moved to test : {moved}")
    print(f"  val tiles remaining : {remaining}")


# ---------------------------------------------------------------------------
# Step 4 — Synthetic dataset
# ---------------------------------------------------------------------------

def step_synthetic(synth_n: int, seed: int, force: bool) -> None:
    print(f"\n{SEP}")
    print(f"Step 4a: Generate synthetic dataset  (n={synth_n}  seed={seed})")
    print(SEP)
    if not force and _has_files(SYNTH / "images/train"):
        print(f"  SKIP: {SYNTH}/images/train already has sheets")
        return
    if force and SYNTH.exists():
        shutil.rmtree(SYNTH)

    glyphs = build_glyph_library(SRC / "images", SRC / "labels")
    missing = [c for c in range(NC) if not glyphs.get(c)]
    if missing:
        print(f"  WARNING: classes with no glyphs: {missing}")
    total_crops = sum(len(v) for v in glyphs.values())
    print(f"  glyph library: {total_crops} crops across {len(glyphs)} / {NC} classes")

    generate_dataset(glyphs, n=synth_n, out_dir=SYNTH, seed=seed)


def step_tile_synthetic(force: bool) -> None:
    print(f"\n{SEP}")
    print("Step 4b: Tile synthetic sheets  (640px / 20% overlap)")
    print(SEP)
    if not force and _has_files(SYNTH_TILED / "images/train"):
        print(f"  SKIP: {SYNTH_TILED}/images/train already has tiles")
        return
    if force and SYNTH_TILED.exists():
        shutil.rmtree(SYNTH_TILED)

    print(f"  Source: {SYNTH}")
    results = slice_dataset(SYNTH / "images", SYNTH / "labels", SYNTH_TILED)
    for split, s in results.items():
        ret = s["boxes_out"] / max(s["boxes_in"], 1)
        print(f"  [{split}]  {s['tiles_written']:>6} tiles  "
              f"{s['boxes_out']:>7} boxes  (retention {ret:.3f}x)")


# ---------------------------------------------------------------------------
# Step 5 — Merge into data/merged/
# ---------------------------------------------------------------------------

def step_merge(force: bool) -> None:
    print(f"\n{SEP}")
    print("Step 5: Merge into data/merged/  (hard links, copy fallback)")
    print(SEP)
    if force and MERGED.exists():
        shutil.rmtree(MERGED)

    # Synthetic TILED tiles go into train ONLY — val and test stay real-only
    img_sources = {
        "train": [TILED / "images/train", SYNTH_TILED / "images/train"],
        "val":   [TILED / "images/val"],
        "test":  [TILED / "images/test"],
    }
    lbl_sources = {
        "train": [TILED / "labels/train", SYNTH_TILED / "labels/train"],
        "val":   [TILED / "labels/val"],
        "test":  [TILED / "labels/test"],
    }

    for split in ("train", "val", "test"):
        img_dst = MERGED / "images" / split
        lbl_dst = MERGED / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_imgs = n_lbls = 0
        for src_dir in img_sources[split]:
            for f in sorted(src_dir.glob("*.jpg")):
                _link_or_copy(f, img_dst / f.name)
                n_imgs += 1
        for src_dir in lbl_sources[split]:
            for f in sorted(src_dir.glob("*.txt")):
                _link_or_copy(f, lbl_dst / f.name)
                n_lbls += 1
        print(f"  [{split}]  {n_imgs:>6} images  {n_lbls:>6} labels")


# ---------------------------------------------------------------------------
# Step 6 — Write dataset YAML
# ---------------------------------------------------------------------------

def step_write_yaml() -> None:
    print(f"\n{SEP}")
    print("Step 6: Update configs/yolo_baseline.yaml")
    print(SEP)

    with open(YAML) as fh:
        cfg = yaml.safe_load(fh)
    names = cfg["names"]
    nc    = cfg.get("nc", NC)

    lines = [
        "# Auto-generated by scripts/build_dataset.py -- edit the script, not this file.",
        "path: data/merged",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for i in range(nc):
        lines.append(f"  {i}: {names[i]}")

    YAML.write_text("\n".join(lines) + "\n")
    print(f"  Written {YAML}")
    print(f"  path=data/merged   nc={nc}   splits=train/val/test")


# ---------------------------------------------------------------------------
# Step 7 — Per-class histogram
# ---------------------------------------------------------------------------

def step_histogram() -> None:
    print(f"\n{SEP}")
    print("Step 7: Per-class instance counts  (merged train)")
    print(SEP)
    counts: Counter = Counter()
    for lbl in sorted((MERGED / "labels/train").glob("*.txt")):
        for line in lbl.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1

    total = sum(counts.values())
    max_c = max(counts.values()) if counts else 1
    min_c = min(counts.values()) if counts else 0
    print(f"  total train boxes : {total:,}")
    print(f"  max / min per cls : {max_c:,} / {min_c:,}  "
          f"(imbalance {max_c / max(min_c, 1):.1f}x)")
    print()
    for cls in range(NC):
        n   = counts.get(cls, 0)
        bar = "#" * max(1, round(n * 40 / max_c))
        print(f"    cls {cls:>2}: {n:>6}  {bar}")

    hist_out = Path("data/merged_class_distribution.png")
    save_class_distribution(counts, NC, hist_out)
    print(f"\n  histogram -> {hist_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",          type=int,   default=42,
                    help="master seed for test-carve shuffle and synthetic generation")
    ap.add_argument("--synth-n",       type=int,   default=200,
                    help="number of synthetic sheets to generate")
    ap.add_argument("--test-fraction", type=float, default=0.20,
                    help="fraction of HF val source images reserved for test")
    ap.add_argument("--force",         action="store_true",
                    help="clear and re-run all steps from scratch")
    args = ap.parse_args()

    print(f"\n{SEP}")
    print("PIDetect -- Phase 0 one-command dataset build")
    print(f"  seed={args.seed}  synth_n={args.synth_n}  "
          f"test_fraction={args.test_fraction}  force={args.force}")
    print(SEP)

    step_download(args.force)
    step_tile(args.force)
    step_carve_test(args.seed, args.test_fraction, args.force)
    step_synthetic(args.synth_n, args.seed, args.force)
    step_tile_synthetic(args.force)
    step_merge(args.force)
    step_write_yaml()
    step_histogram()

    # Summary
    print(f"\n{SEP}")
    print("Phase 0 COMPLETE")
    for split in ("train", "val", "test"):
        n = len(list((MERGED / "images" / split).glob("*.jpg")))
        real = "real+synth" if split == "train" else "real only"
        print(f"  {split:<5}: {n:>6} tiles  ({real})")
    print(f"\n  Config   : {YAML}")
    print(f"  Histogram: data/merged_class_distribution.png")
    print(f"  Rebuild  : python scripts/build_dataset.py")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
