"""Subtask 1.7d — before/after synth preview: arrow size + valve variety.

Generates docs/synth_preview.png showing:
  Row 1: ARROWS — before (sampled from existing data/synthetic/)
  Row 2: ARROWS — after  (new synth.py, log-uniform 12–80 px)
  Row 3: VALVES — after  (new synth.py, with stem/actuator/flange variety)

Also prints arrow diagonal stats for old vs new.

Usage (from project root):
    python scripts/preview_synth.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pidetect.data.synth import (
    ARROW_IDX, VALVE_IDX,
    build_glyph_library, compose_sheet,
)

SRC   = Path("data/digitize-pid-yolo/DigitizePID_Dataset")
OLD_SYNTH_IMAGES = Path("data/synthetic/images/train")
OLD_SYNTH_LABELS = Path("data/synthetic/labels/train")
OUT   = Path("docs/synth_preview.png")

CROP_SIZE   = 96    # each thumbnail cell (px)
N_CROPS      = 8    # thumbnails per row
N_NEW_SHEETS = 100  # sheets to generate for the "after" sample pool
N_STAT_OLD   = 200  # max arrow instances to collect from old synth for stats


def _diag(w: int, h: int) -> float:
    return math.sqrt(w * w + h * h)


def _thumb(crop: np.ndarray) -> np.ndarray:
    """Resize crop to CROP_SIZE×CROP_SIZE on a white background."""
    h, w = crop.shape[:2]
    scale = CROP_SIZE / max(h, w, 1)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    canvas = np.full((CROP_SIZE, CROP_SIZE, 3), 255, dtype=np.uint8)
    y0 = (CROP_SIZE - th) // 2
    x0 = (CROP_SIZE - tw) // 2
    canvas[y0: y0 + th, x0: x0 + tw] = small
    return canvas


def collect_old_arrow_crops(
    n_display: int = N_CROPS,
    n_stat: int = N_STAT_OLD,
) -> tuple[list[np.ndarray], list[float]]:
    """Sample arrow (class 23) crops from existing data/synthetic/.

    Returns (display_crops[:n_display], all_diags[:n_stat]) so stats use
    more samples than the thumbnail row.
    """
    crops, diags = [], []
    if not OLD_SYNTH_IMAGES.exists():
        return crops, diags
    rng = np.random.default_rng(0)
    img_paths = sorted(OLD_SYNTH_IMAGES.glob("*.jpg"))
    rng.shuffle(img_paths)
    for img_path in img_paths:
        if len(diags) >= n_stat:
            break
        lbl_path = OLD_SYNTH_LABELS / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if int(parts[0]) != ARROW_IDX:
                continue
            xc, yc, bw, bh = (float(p) for p in parts[1:5])
            x1 = int((xc - bw / 2) * iw)
            y1 = int((yc - bh / 2) * ih)
            x2 = int((xc + bw / 2) * iw)
            y2 = int((yc + bh / 2) * ih)
            crop = img[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
            if crop.size > 0:
                if len(crops) < n_display:
                    crops.append(crop)
                diags.append(_diag(x2 - x1, y2 - y1))
            if len(diags) >= n_stat:
                break
    return crops, diags


def collect_new_crops(
    glyphs: dict,
    n_sheets: int = N_NEW_SHEETS,
    n_display: int = N_CROPS,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    """Generate new sheets and collect arrow + valve crops.

    Collects ALL arrow diagonals across n_sheets for stable stats,
    but only keeps n_display thumbnails for the visual row.
    """
    arrow_crops, arrow_diags, valve_crops = [], [], []
    rng = np.random.default_rng(1)

    for i in range(n_sheets):
        if (i + 1) % 20 == 0:
            print(f"  … {i + 1}/{n_sheets} sheets generated")
        img, yolo_boxes, placed, _ = compose_sheet(glyphs, n_symbols=40, rng=rng)
        ih, iw = img.shape[:2]
        for box in yolo_boxes:
            xc, yc, bw, bh = box["xc"], box["yc"], box["w"], box["h"]
            x1 = int((xc - bw / 2) * iw)
            y1 = int((yc - bh / 2) * ih)
            x2 = int((xc + bw / 2) * iw)
            y2 = int((yc + bh / 2) * ih)
            crop = img[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
            if crop.size == 0:
                continue
            if box["cls"] == ARROW_IDX:
                arrow_diags.append(_diag(x2 - x1, y2 - y1))
                if len(arrow_crops) < n_display:
                    arrow_crops.append(crop)
            elif box["cls"] in VALVE_IDX and len(valve_crops) < n_display:
                valve_crops.append(crop)

    return arrow_crops, arrow_diags, valve_crops


def make_row(crops: list[np.ndarray], label: str, n: int = N_CROPS) -> np.ndarray:
    """Build a single labeled row of thumbnail cells."""
    PAD = 4
    LABEL_H = 20
    cell_w = CROP_SIZE + PAD
    row_w = cell_w * n + PAD
    row_h = CROP_SIZE + PAD * 2 + LABEL_H
    row = np.full((row_h, row_w, 3), 230, dtype=np.uint8)

    # label
    cv2.putText(row, label, (PAD, LABEL_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    # thumbnails
    for i, crop in enumerate(crops[:n]):
        x0 = PAD + i * cell_w
        y0 = LABEL_H + PAD
        thumb = _thumb(crop)
        row[y0: y0 + CROP_SIZE, x0: x0 + CROP_SIZE] = thumb

    return row


def main() -> None:
    print("Building glyph library …")
    if not SRC.exists():
        sys.exit(f"[error] HF dataset not found at {SRC}. Run build_dataset.py first.")
    glyphs = build_glyph_library(SRC / "images", SRC / "labels")

    print("Collecting OLD arrow crops from data/synthetic/ …")
    old_arrow_crops, old_diags = collect_old_arrow_crops()

    print(f"Generating {N_NEW_SHEETS} new sheets for 'after' crops …")
    new_arrow_crops, new_diags, new_valve_crops = collect_new_crops(glyphs)

    # ── stats ────────────────────────────────────────────────────────────────
    def _stats(diags: list[float]) -> str:
        if not diags:
            return "n/a"
        arr = np.array(diags)
        return (f"n={len(arr)}  median={np.median(arr):.1f}px  "
                f"mean={arr.mean():.1f}px  "
                f"range=[{arr.min():.1f}, {arr.max():.1f}]")

    print("\n=== Arrow diagonal stats ===")
    print(f"  BEFORE (old synth): {_stats(old_diags)}")
    print(f"  AFTER  (1.7d synth): {_stats(new_diags)}")

    # ── montage ──────────────────────────────────────────────────────────────
    rows = [
        make_row(old_arrow_crops,  "ARROWS — before (old synth, median ~79 px)"),
        make_row(new_arrow_crops,  "ARROWS — after  (1.7d, log-uniform 12–80 px)"),
        make_row(new_valve_crops,  "VALVES — after  (1.7d, stem/actuator/flange variety)"),
    ]

    # pad all rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 230, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    montage = np.vstack(padded)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), montage)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
