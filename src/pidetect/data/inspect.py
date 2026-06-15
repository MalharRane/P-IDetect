"""Sanity-check the dataset BEFORE training: per-class counts + sample box overlays.

This is the honesty step. If a class has ~no instances, you'll see it here, not after
wasting GPU hours. Run after download.

Usage:
    python -m src.pidetect.data.inspect --root data/digitize-pid-yolo
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def count_classes(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for txt in labels_dir.rglob("*.txt"):
        for line in txt.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def draw_sample(img_path: Path, label_path: Path, out: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = (float(v) for v in line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, str(int(cls)), (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/digitize-pid-yolo")
    args = ap.parse_args()
    root = Path(args.root)

    labels = root.rglob("*.txt")
    counts = count_classes(root)
    print("Per-class instance counts (watch for tiny classes):")
    for cls, n in sorted(counts.items()):
        print(f"  class {cls:>3}: {n}")
    print(f"  total boxes: {sum(counts.values())}  |  classes: {len(counts)}")

    # plot distribution
    if counts:
        plt.figure(figsize=(10, 4))
        plt.bar([str(c) for c in sorted(counts)], [counts[c] for c in sorted(counts)])
        plt.title("Class distribution"); plt.xlabel("class"); plt.ylabel("instances")
        plt.tight_layout(); plt.savefig("data/class_distribution.png", dpi=120)
        print("Saved data/class_distribution.png")

    # TODO: draw a few sample overlays into data/samples/ using draw_sample()
    print("TODO: render a few box overlays to eyeball annotation quality.")


if __name__ == "__main__":
    main()
