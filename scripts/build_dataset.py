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
NC           = 32
SEP          = "=" * 62


def _paths(tile: int) -> tuple[Path, Path, Path, Path, Path]:
    """Return (TILED, SYNTH, SYNTH_TILED, MERGED, YAML) for a given tile size.

    tile=640 → current canonical paths (backward-compatible).
    tile=320 → parallel paths so the 640px dataset is not overwritten.
    """
    suffix = "" if tile == 640 else f"_{tile}"
    tiled       = Path(f"data/tiled{suffix}")
    synth       = Path("data/synthetic")          # synthetic sheets are tile-independent
    synth_tiled = Path(f"data/synthetic_tiled{suffix}")
    merged      = Path(f"data/merged{suffix}")
    yaml_path   = Path("configs/yolo_baseline.yaml" if tile == 640
                       else f"configs/yolo_{tile}.yaml")
    return tiled, synth, synth_tiled, merged, yaml_path


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

def step_tile(force: bool, tiled: Path, tile: int) -> None:
    print(f"\n{SEP}")
    print(f"Step 2: Tile real dataset ({tile}px / 20% overlap)")
    print(SEP)
    if not force and _has_files(tiled / "images/train"):
        print(f"  SKIP: {tiled}/images/train already has tiles")
        return
    if force:
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = tiled / sub / split
                if d.exists():
                    shutil.rmtree(d)

    print(f"  Source: {SRC}")
    results = slice_dataset(SRC / "images", SRC / "labels", tiled, tile=tile)
    for split, s in results.items():
        ret = s["boxes_out"] / max(s["boxes_in"], 1)
        print(f"  [{split}]  {s['tiles_written']:>6} tiles  "
              f"{s['boxes_out']:>7} boxes  (retention {ret:.3f}x)")


# ---------------------------------------------------------------------------
# Step 3 — Carve test split from tiled val (at source-image granularity)
# ---------------------------------------------------------------------------

def step_carve_test(seed: int, test_fraction: float, force: bool, tiled: Path) -> None:
    print(f"\n{SEP}")
    print("Step 3: Carve test split from tiled val  (real data only)")
    print(SEP)
    if not force and _has_files(tiled / "images/test"):
        print(f"  SKIP: {tiled}/images/test already has tiles")
        return

    val_tiles = sorted((tiled / "images/val").glob("*.jpg"))
    if not val_tiles:
        raise RuntimeError(
            f"No tiles in {tiled}/images/val -- step 2 must run first."
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

    (tiled / "images/test").mkdir(parents=True, exist_ok=True)
    (tiled / "labels/test").mkdir(parents=True, exist_ok=True)

    moved = 0
    for t in val_tiles:
        if t.stem.rsplit("_", 1)[0] in test_set:
            t.rename(tiled / "images/test" / t.name)
            lbl = tiled / "labels/val" / (t.stem + ".txt")
            if lbl.exists():
                lbl.rename(tiled / "labels/test" / lbl.name)
            moved += 1

    remaining = len(list((tiled / "images/val").glob("*.jpg")))
    print(f"  tiles moved to test : {moved}")
    print(f"  val tiles remaining : {remaining}")


# ---------------------------------------------------------------------------
# Step 4 — Synthetic dataset
# ---------------------------------------------------------------------------

def step_synthetic(synth_n: int, seed: int, force: bool, synth: Path) -> None:
    print(f"\n{SEP}")
    print(f"Step 4a: Generate synthetic dataset  (n={synth_n}  seed={seed})")
    print(SEP)
    if not force and _has_files(synth / "images/train"):
        print(f"  SKIP: {synth}/images/train already has sheets")
        return
    if force and synth.exists():
        shutil.rmtree(synth)

    glyphs = build_glyph_library(SRC / "images", SRC / "labels")
    missing = [c for c in range(NC) if not glyphs.get(c)]
    if missing:
        print(f"  WARNING: classes with no glyphs: {missing}")
    total_crops = sum(len(v) for v in glyphs.values())
    print(f"  glyph library: {total_crops} crops across {len(glyphs)} / {NC} classes")

    generate_dataset(glyphs, n=synth_n, out_dir=synth, seed=seed)


def step_tile_synthetic(force: bool, synth: Path, synth_tiled: Path, tile: int) -> None:
    print(f"\n{SEP}")
    print(f"Step 4b: Tile synthetic sheets  ({tile}px / 20% overlap)")
    print(SEP)
    if not force and _has_files(synth_tiled / "images/train"):
        print(f"  SKIP: {synth_tiled}/images/train already has tiles")
        return
    if force and synth_tiled.exists():
        shutil.rmtree(synth_tiled)

    print(f"  Source: {synth}")
    results = slice_dataset(synth / "images", synth / "labels", synth_tiled, tile=tile)
    for split, s in results.items():
        ret = s["boxes_out"] / max(s["boxes_in"], 1)
        print(f"  [{split}]  {s['tiles_written']:>6} tiles  "
              f"{s['boxes_out']:>7} boxes  (retention {ret:.3f}x)")


# ---------------------------------------------------------------------------
# Step 5 — Merge into data/merged/
# ---------------------------------------------------------------------------

def step_merge(force: bool, tiled: Path, synth_tiled: Path, merged: Path) -> None:
    print(f"\n{SEP}")
    print(f"Step 5: Merge into {merged}/  (hard links, copy fallback)")
    print(SEP)
    if force and merged.exists():
        shutil.rmtree(merged)

    # Synthetic TILED tiles go into train ONLY — val and test stay real-only
    img_sources = {
        "train": [tiled / "images/train", synth_tiled / "images/train"],
        "val":   [tiled / "images/val"],
        "test":  [tiled / "images/test"],
    }
    lbl_sources = {
        "train": [tiled / "labels/train", synth_tiled / "labels/train"],
        "val":   [tiled / "labels/val"],
        "test":  [tiled / "labels/test"],
    }

    for split in ("train", "val", "test"):
        img_dst = merged / "images" / split
        lbl_dst = merged / "labels" / split
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

def step_write_yaml(merged: Path, yaml_path: Path) -> None:
    print(f"\n{SEP}")
    print(f"Step 6: Update {yaml_path}")
    print(SEP)

    # Read names from the canonical baseline yaml (always present after first build)
    baseline = Path("configs/yolo_baseline.yaml")
    src_yaml = baseline if baseline.exists() else yaml_path
    with open(src_yaml) as fh:
        cfg = yaml.safe_load(fh)
    names = cfg["names"]
    nc    = cfg.get("nc", NC)

    lines = [
        "# Auto-generated by scripts/build_dataset.py -- edit the script, not this file.",
        f"path: {merged}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for i in range(nc):
        lines.append(f"  {i}: {names[i]}")

    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"  Written {yaml_path}")
    print(f"  path={merged}   nc={nc}   splits=train/val/test")


# ---------------------------------------------------------------------------
# Step 7 — Per-class histogram
# ---------------------------------------------------------------------------

def step_histogram(merged: Path) -> None:
    print(f"\n{SEP}")
    print(f"Step 7: Per-class instance counts  ({merged}/labels/train)")
    print(SEP)
    counts: Counter = Counter()
    for lbl in sorted((merged / "labels/train").glob("*.txt")):
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

    stem = merged.name  # "merged" or "merged_320"
    hist_out = Path(f"data/{stem}_class_distribution.png")
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
    ap.add_argument("--tile",          type=int,   default=640,
                    help="tile side length in px (default 640). Use 320 for phase 1.8c "
                         "higher-resolution training. Non-640 values write to separate "
                         "data/tiled_<N>, data/merged_<N> and configs/yolo_<N>.yaml so "
                         "the canonical 640px dataset is not overwritten.")
    ap.add_argument("--force",         action="store_true",
                    help="clear and re-run all steps from scratch")
    args = ap.parse_args()

    tiled, synth, synth_tiled, merged, yaml_path = _paths(args.tile)

    print(f"\n{SEP}")
    print("PIDetect -- Phase 0 one-command dataset build")
    print(f"  seed={args.seed}  synth_n={args.synth_n}  tile={args.tile}px  "
          f"test_fraction={args.test_fraction}  force={args.force}")
    print(f"  -> tiled:      {tiled}")
    print(f"  -> merged:     {merged}")
    print(f"  -> yaml:       {yaml_path}")
    print(SEP)

    step_download(args.force)
    step_tile(args.force, tiled, args.tile)
    step_carve_test(args.seed, args.test_fraction, args.force, tiled)
    step_synthetic(args.synth_n, args.seed, args.force, synth)
    step_tile_synthetic(args.force, synth, synth_tiled, args.tile)
    step_merge(args.force, tiled, synth_tiled, merged)
    step_write_yaml(merged, yaml_path)
    step_histogram(merged)

    # Summary
    print(f"\n{SEP}")
    print("Phase 0 COMPLETE")
    for split in ("train", "val", "test"):
        n = len(list((merged / "images" / split).glob("*.jpg")))
        real = "real+synth" if split == "train" else "real only"
        print(f"  {split:<5}: {n:>6} tiles  ({real})")
    print(f"\n  Config   : {yaml_path}")
    print(f"  Histogram: data/merged_class_distribution.png")
    print(f"  Rebuild  : python scripts/build_dataset.py --tile {args.tile}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
