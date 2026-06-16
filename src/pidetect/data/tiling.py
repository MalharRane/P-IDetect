"""SAHI-style tiling: slice large P&IDs into overlapping tiles for train & inference.

Full sheets are ~7000 x 4500 px; a 640-px model sees nothing without tiling.
EDA (subtask 0.3) found median symbol = 72 px, p90 = 127 px on these images.
A 640-px tile with 20% overlap gives ~5x context around the median symbol and
comfortably contains the p90 box.

Usage:
    python -m src.pidetect.data.tiling
    python -m src.pidetect.data.tiling --tile 640 --overlap 0.2 --neg-fraction 0.05
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def slice_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    tile: int = 640,
    overlap: float = 0.2,
    neg_fraction: float = 0.0,
) -> dict[str, dict]:
    """Tile every split (train/val) found under images_dir and labels_dir.

    Parameters
    ----------
    images_dir:   root containing train/ and val/ sub-folders of images.
    labels_dir:   root containing matching train/ and val/ label folders.
    out_dir:      destination root; receives images/{train,val} and
                  labels/{train,val} in standard YOLO layout.
    tile:         tile side length in pixels (square).
    overlap:      fractional overlap between adjacent tiles (0..1).
    neg_fraction: fraction of empty tiles to keep as hard-negative examples.

    Returns
    -------
    dict mapping split name -> stats dict with keys:
        tiles_written, boxes_in, boxes_out, classes_seen
    """
    results = {}
    for split in ("train", "val"):
        img_split = images_dir / split
        lbl_split = labels_dir / split
        if not img_split.exists():
            continue
        out_img = out_dir / "images" / split
        out_lbl = out_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        stats: dict = {"tiles_written": 0, "boxes_in": 0, "boxes_out": 0,
                       "classes_seen": set()}
        for img_path in sorted(img_split.glob("*.jpg")):
            lbl_path = lbl_split / (img_path.stem + ".txt")
            split_stats = _slice_image(
                img_path, lbl_path, out_img, out_lbl,
                tile=tile, overlap=overlap, neg_fraction=neg_fraction,
            )
            stats["tiles_written"] += split_stats["tiles_written"]
            stats["boxes_in"]      += split_stats["boxes_in"]
            stats["boxes_out"]     += split_stats["boxes_out"]
            stats["classes_seen"].update(split_stats["classes_seen"])
        results[split] = stats
    return results


def merge_tile_predictions(
    tile_preds: list[dict],
    orig_width: int,
    orig_height: int,
    tile: int = 640,
    overlap: float = 0.2,
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Merge per-tile detections back into full-image coordinates (Phase 1).

    Each entry in tile_preds must contain:
        row (int), col (int): tile grid position
        boxes (list[dict]): each with keys cls, xc, yc, w, h in pixel coords
                            relative to the tile origin.

    After translating tile-local boxes to full-image coords the function
    removes cross-tile duplicates with NMS at the given iou_threshold.

    NOTE: In Phase 1 this will be handled end-to-end by
    sahi.get_sliced_prediction; this stub is the seam that Phase 4
    evaluation code will call for graph-building.
    """
    raise NotImplementedError(
        "Phase 1: implement via sahi.get_sliced_prediction or manual NMS."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tile_positions(img_w: int, img_h: int, tile: int, overlap: float
                    ) -> list[tuple[int, int, int, int]]:
    """Return (wx1, wy1, wx2, wy2) for every tile window.

    Tiles are always exactly tile x tile px. The last tile in each row/column
    is shifted left/up so it does not exceed the image boundary (it therefore
    overlaps more than `overlap` with the penultimate tile). For images smaller
    than `tile` on an axis a single tile covering the whole axis is returned.
    """
    step = max(1, int(tile * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        pts = list(range(0, length - tile, step))
        if not pts or pts[-1] + tile < length:
            pts.append(length - tile)
        return pts

    windows = []
    for y in starts(img_h):
        for x in starts(img_w):
            windows.append((x, y, x + tile, y + tile))
    return windows


def _read_labels_px(lbl_path: Path, img_w: int, img_h: int
                    ) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file; return absolute-pixel (cls, x1, y1, x2, y2)."""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _area_overlap_fraction(bx1: float, by1: float, bx2: float, by2: float,
                           wx1: int, wy1: int, wx2: int, wy2: int) -> float:
    """Fraction of the box area that lies inside the window."""
    ix1, iy1 = max(bx1, wx1), max(by1, wy1)
    ix2, iy2 = min(bx2, wx2), min(by2, wy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    box_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return inter / box_area


def slice_image(
    img_path: Path,
    lbl_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile: int = 640,
    overlap: float = 0.2,
    neg_fraction: float = 0.0,
) -> dict:
    """Tile a single image/label pair (public wrapper around the same logic
    `slice_dataset` uses per-image). For datasets that don't have a train/val
    directory structure -- e.g. the real-world eval tiers (subtask 1.6c),
    which tile one source sheet or a handful of standalone sheets at a time.
    """
    return _slice_image(img_path, lbl_path, out_img_dir, out_lbl_dir,
                        tile, overlap, neg_fraction)


def _slice_image(
    img_path: Path,
    lbl_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile: int,
    overlap: float,
    neg_fraction: float,
) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"tiles_written": 0, "boxes_in": 0, "boxes_out": 0,
                "classes_seen": set()}

    img_h, img_w = img.shape[:2]
    src_boxes = _read_labels_px(lbl_path, img_w, img_h)
    windows   = _tile_positions(img_w, img_h, tile, overlap)

    tiles_written   = 0
    boxes_out_total = 0
    classes_seen: set[int] = set()

    for row, (wx1, wy1, wx2, wy2) in enumerate(windows):
        tw = wx2 - wx1
        th = wy2 - wy1
        tile_lines: list[str] = []

        for cls, bx1, by1, bx2, by2 in src_boxes:
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            center_in    = (wx1 <= bcx < wx2) and (wy1 <= bcy < wy2)
            overlap_frac = _area_overlap_fraction(bx1, by1, bx2, by2,
                                                  wx1, wy1, wx2, wy2)
            if not (center_in or overlap_frac >= 0.4):
                continue

            # clip to window, renormalise to tile
            cx1 = max(bx1, wx1);  cy1 = max(by1, wy1)
            cx2 = min(bx2, wx2);  cy2 = min(by2, wy2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            new_xc = ((cx1 + cx2) / 2 - wx1) / tw
            new_yc = ((cy1 + cy2) / 2 - wy1) / th
            new_w  = (cx2 - cx1) / tw
            new_h  = (cy2 - cy1) / th
            tile_lines.append(
                f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
            )
            classes_seen.add(cls)

        if not tile_lines:
            if neg_fraction <= 0.0 or random.random() >= neg_fraction:
                continue

        stem = f"{img_path.stem}_{row:05d}"
        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), img[wy1:wy2, wx1:wx2])
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(tile_lines))

        tiles_written   += 1
        boxes_out_total += len(tile_lines)

    return {
        "tiles_written": tiles_written,
        "boxes_in":      len(src_boxes),
        "boxes_out":     boxes_out_total,
        "classes_seen":  classes_seen,
    }


def _visualize_samples(
    tiled_img_dir: Path,
    tiled_lbl_dir: Path,
    out_dir: Path,
    n: int = 5,
) -> None:
    """Draw remapped boxes on evenly-spaced sample tiles and write to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles  = sorted(tiled_img_dir.glob("*.jpg"))
    step   = max(1, len(tiles) // n)
    chosen = tiles[::step][:n]
    for img_path in chosen:
        lbl_path = tiled_lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                xc, yc, bw, bh = (float(p) for p in parts[1:5])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 60, 220), 2)
                cv2.putText(img, str(cls), (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 220), 1,
                            cv2.LINE_AA)
        cv2.imwrite(str(out_dir / img_path.name), img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root",     default="data/digitize-pid-yolo/DigitizePID_Dataset",
                    help="source dataset root (contains images/ and labels/)")
    ap.add_argument("--out",          default="data/tiled",
                    help="output directory for tiled YOLO dataset")
    ap.add_argument("--tile",         type=int,   default=640)
    ap.add_argument("--overlap",      type=float, default=0.2)
    ap.add_argument("--neg-fraction", type=float, default=0.0,
                    help="fraction of empty tiles kept as negatives")
    ap.add_argument("--samples",      type=int,   default=5,
                    help="number of sample tiles to visualise")
    args = ap.parse_args()

    src = Path(args.src_root)
    out = Path(args.out)
    SEP = "=" * 62

    print(f"\n{SEP}")
    print(f"Tiling  {src}")
    print(f"  tile={args.tile}  overlap={args.overlap}"
          f"  neg_fraction={args.neg_fraction}")
    print(f"  -> {out}")
    print(SEP)

    results = slice_dataset(
        images_dir=src / "images",
        labels_dir=src / "labels",
        out_dir=out,
        tile=args.tile,
        overlap=args.overlap,
        neg_fraction=args.neg_fraction,
    )

    total_tiles = 0
    for split, s in results.items():
        retention = s["boxes_out"] / max(s["boxes_in"], 1)
        print(f"\n[{split}]")
        print(f"  tiles written : {s['tiles_written']:,}")
        print(f"  boxes in      : {s['boxes_in']:,}")
        print(f"  boxes out     : {s['boxes_out']:,}  ({retention:.3f} retention)")
        print(f"  classes seen  : {len(s['classes_seen'])}"
              f"  (first 8: {sorted(s['classes_seen'])[:8]})")
        total_tiles += s["tiles_written"]

    print(f"\n  TOTAL tiles   : {total_tiles:,}")

    # sample overlays
    tiled_img = out / "images" / "train"
    tiled_lbl = out / "labels" / "train"
    if tiled_img.exists():
        samples_dir = Path("data/tiled_samples")
        _visualize_samples(tiled_img, tiled_lbl, samples_dir, n=args.samples)
        print(f"  {args.samples} sample tiles -> {samples_dir}/")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
