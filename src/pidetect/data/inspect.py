"""Sanity-check the dataset BEFORE training: per-class counts, image stats,
box-size distribution, and sample bounding-box overlays.

This is the honesty step. If a class has ~no instances, or boxes are so tiny
that a 640-px tile would miss them, you'll see it here -- not after wasting GPU
hours. The box-size stats directly drive tile-size selection in subtask 0.4.

Run after download:
    python -m src.pidetect.data.inspect
    python -m src.pidetect.data.inspect --root data/digitize-pid-yolo/DigitizePID_Dataset
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# -- low-level readers ----------------------------------------------------------

def _label_rows(labels_dir: Path) -> list[tuple[int, float, float, float, float]]:
    """Read every YOLO label row under labels_dir into (cls, xc, yc, bw, bh)."""
    rows: list[tuple[int, float, float, float, float]] = []
    for txt in sorted(labels_dir.rglob("*.txt")):
        for line in txt.read_text().splitlines():
            if line.strip():
                parts = line.split()
                rows.append((int(parts[0]),
                             float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])))
    return rows


def count_classes(labels_dir: Path) -> Counter:
    """Per-class instance count across all splits under labels_dir."""
    counts: Counter = Counter()
    for cls, *_ in _label_rows(labels_dir):
        counts[cls] += 1
    return counts


# -- metric collectors ----------------------------------------------------------

def image_resolution_stats(images_dir: Path) -> dict:
    """Min / median / max width and height over all images."""
    widths, heights = [], []
    for p in sorted(images_dir.rglob("*.jpg")):
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
    return {
        "count": len(widths),
        "width":  (min(widths),  statistics.median(widths),  max(widths)),
        "height": (min(heights), statistics.median(heights), max(heights)),
    }


def boxes_per_image_stats(labels_dir: Path) -> dict:
    """Min / median / max annotation count per label file."""
    counts = []
    for txt in sorted(labels_dir.rglob("*.txt")):
        n = sum(1 for ln in txt.read_text().splitlines() if ln.strip())
        counts.append(n)
    return {
        "min":    min(counts),
        "median": statistics.median(counts),
        "max":    max(counts),
    }


def pixel_box_sizes(images_dir: Path, labels_dir: Path) -> dict:
    """
    Convert normalised YOLO box dims to actual pixel sizes and collect
    distribution stats.  This is the primary input for tile-size selection:
    tile must comfortably contain the p90 box size.
    """
    px_w: list[float] = []
    px_h: list[float] = []
    for img_path in sorted(images_dir.rglob("*.jpg")):
        rel = img_path.relative_to(images_dir)          # e.g. train/0.jpg
        label_path = labels_dir / rel.with_suffix(".txt")
        if not label_path.exists():
            continue
        with Image.open(img_path) as im:
            iw, ih = im.size
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            px_w.append(float(parts[3]) * iw)
            px_h.append(float(parts[4]) * ih)
    arr_w = np.array(px_w)
    arr_h = np.array(px_h)
    return {
        "n_boxes":       len(arr_w),
        "median_w":      float(np.median(arr_w)),
        "median_h":      float(np.median(arr_h)),
        "p10_w":         float(np.percentile(arr_w, 10)),
        "p90_w":         float(np.percentile(arr_w, 90)),
        "p10_h":         float(np.percentile(arr_h, 10)),
        "p90_h":         float(np.percentile(arr_h, 90)),
        "max_w":         float(arr_w.max()),
        "max_h":         float(arr_h.max()),
        "all_w":         px_w,
        "all_h":         px_h,
    }


# -- visualisations -------------------------------------------------------------

def save_class_distribution(counts: Counter, nc: int, out: Path) -> None:
    classes = list(range(nc))
    values  = [counts.get(c, 0) for c in classes]
    fig, ax = plt.subplots(figsize=(16, 4))
    bars = ax.bar([str(c) for c in classes], values, color="steelblue")
    ax.bar_label(bars, fmt="%d", fontsize=6, padding=2)
    max_v = max(values) if values else 1
    ax.axhline(max_v / 10, color="red", linestyle="--", linewidth=0.8,
               label="10% of most-common class")
    ax.set_title("Per-class instance count (train + val)")
    ax.set_xlabel("class index")
    ax.set_ylabel("instances")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out), dpi=130)
    plt.close()


def save_box_size_histogram(px_w: list[float], px_h: list[float], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, label, colour in [
        (ax1, px_w, "Box width (px)",  "steelblue"),
        (ax2, px_h, "Box height (px)", "darkorange"),
    ]:
        ax.hist(data, bins=80, color=colour, edgecolor="none", alpha=0.85)
        med = statistics.median(data)
        p90 = float(np.percentile(data, 90))
        ax.axvline(med, color="black", linewidth=1.2, label=f"median {med:.0f}")
        ax.axvline(p90, color="red",   linewidth=1.2, linestyle="--",
                   label=f"p90 {p90:.0f}")
        ax.set_title(label)
        ax.set_xlabel("pixels")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    plt.suptitle("Symbol bounding-box pixel sizes  <- tile-size driver", y=1.01)
    plt.tight_layout()
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()


def draw_sample(img_path: Path, label_path: Path, out: Path) -> None:
    """Render YOLO boxes on the image and write to out."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts  = line.split()
        cls    = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 60, 220), 2)
        cv2.putText(img, str(cls), (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 220), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(out), img)


# -- main -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/digitize-pid-yolo/DigitizePID_Dataset",
                    help="dataset root containing images/ and labels/")
    ap.add_argument("--nc", type=int, default=32, help="total number of classes")
    ap.add_argument("--samples", type=int, default=5,
                    help="number of sample overlay images to render")
    args = ap.parse_args()

    root       = Path(args.root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    SEP = "=" * 62

    # 1. Per-class counts ------------------------------------------------------
    counts = count_classes(labels_dir)
    total  = sum(counts.values())
    max_c  = max(counts.values())
    min_c  = min(counts.values())
    print(f"\n{SEP}")
    print("1. Per-class instance counts  (train + val)")
    print(SEP)
    for cls in range(args.nc):
        n   = counts.get(cls, 0)
        bar = "#" * max(1, round(n * 40 / max_c))
        print(f"  {cls:>3}: {n:>6}  {bar}")
    print(f"\n  total boxes : {total:,}")
    print(f"  num classes : {len(counts)} / {args.nc}")
    print(f"  max / min   : {max_c} / {min_c}  (imbalance ratio {max_c / min_c:.1f}x)")

    dist_path = Path("data/class_distribution.png")
    save_class_distribution(counts, args.nc, dist_path)
    print(f"  chart       -> {dist_path}")

    # 2. Image resolution ------------------------------------------------------
    print(f"\n{SEP}")
    print("2. Image resolution")
    print(SEP)
    res = image_resolution_stats(images_dir)
    print(f"  images  : {res['count']}")
    print(f"  width   : min={res['width'][0]}  "
          f"median={res['width'][1]}  max={res['width'][2]}  px")
    print(f"  height  : min={res['height'][0]}  "
          f"median={res['height'][1]}  max={res['height'][2]}  px")

    # 3. Boxes per image -------------------------------------------------------
    print(f"\n{SEP}")
    print("3. Boxes per image")
    print(SEP)
    bpi = boxes_per_image_stats(labels_dir)
    print(f"  min={bpi['min']}  median={bpi['median']}  max={bpi['max']}")

    # 4. Symbol pixel size -----------------------------------------------------
    print(f"\n{SEP}")
    print("4. Symbol bounding-box size in PIXELS  <- tile-size driver")
    print(SEP)
    pbs = pixel_box_sizes(images_dir, labels_dir)
    print(f"  total boxes  : {pbs['n_boxes']:,}")
    print(f"  median       : {pbs['median_w']:.1f} x {pbs['median_h']:.1f} px  (w x h)")
    print(f"  p10 / p90 w  : {pbs['p10_w']:.1f} / {pbs['p90_w']:.1f} px")
    print(f"  p10 / p90 h  : {pbs['p10_h']:.1f} / {pbs['p90_h']:.1f} px")
    print(f"  largest box  : {pbs['max_w']:.1f} x {pbs['max_h']:.1f} px")
    print(f"\n  -> tile should be >= {int(pbs['p90_w'] * 4)} px wide to fit p90 box "
          f"({pbs['p90_w']:.0f} px) with ~4x context")

    hist_path = Path("data/box_size_histogram.png")
    save_box_size_histogram(pbs["all_w"], pbs["all_h"], hist_path)
    print(f"  histogram   -> {hist_path}")

    # 5. Sample overlays -------------------------------------------------------
    print(f"\n{SEP}")
    print(f"5. Sample overlays  (n={args.samples})")
    print(SEP)
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    train_imgs  = sorted((images_dir / "train").glob("*.jpg"))
    # pick 5 evenly spaced to get variety
    step    = max(1, len(train_imgs) // args.samples)
    chosen  = train_imgs[::step][: args.samples]
    for img_path in chosen:
        lbl = labels_dir / "train" / (img_path.stem + ".txt")
        out = samples_dir / f"sample_{img_path.stem}.jpg"
        draw_sample(img_path, lbl, out)
        print(f"  -> {out}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
