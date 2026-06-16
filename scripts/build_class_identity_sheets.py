"""Subtask 1.6a -- build one ground-truth contact sheet per class index.

Crops real instances of every class (0..nc-1) straight from the original
labelled Dataset-P&ID images and tiles them into docs/class_identity/idx_NN.png,
so each index's identity can be verified by eye against the dataset paper's
Symbol1..Symbol32 figure (arXiv 2109.03794, Fig. 3) instead of guessed.

Usage (from project root):
    python scripts/build_class_identity_sheets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pidetect.data.inspect import save_class_identity_sheets

ROOT    = Path("data/digitize-pid-yolo/DigitizePID_Dataset")
OUT_DIR = Path("docs/class_identity")
NC      = 32


def main() -> None:
    stats = save_class_identity_sheets(
        images_dir=ROOT / "images",
        labels_dir=ROOT / "labels",
        out_dir=OUT_DIR,
        nc=NC,
    )
    print(f"{'idx':>3}  {'n':>6}  {'median_w':>9}  {'median_h':>9}")
    for cls in range(NC):
        s = stats[cls]
        print(f"{cls:>3}  {s['n']:>6}  {s['median_w']:>9.1f}  {s['median_h']:>9.1f}")
    print(f"\nWrote {NC} contact sheets -> {OUT_DIR}/idx_NN.png")


if __name__ == "__main__":
    main()
