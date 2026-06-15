"""SAHI sliced full-sheet inference.

Real P&IDs are ~7000×4500 px; training used 640-px tiles. This module slices a
full sheet into overlapping tiles, runs per-tile detection, and merges results
with greedy NMM so cross-seam duplicates are suppressed.

Usage:
    # smoke-test with 1-epoch weights + first HF sheet (CPU, ~2 min):
    python -m pidetect.detect.predict --smoke

    # real inference:
    python -m pidetect.detect.predict \\
        --weights runs/detect/train/weights/best.pt \\
        --image   data/digitize-pid-yolo/DigitizePID_Dataset/images/train/0.jpg \\
        --out     out/predict/sheet0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# One deterministic colour per class index (BGR, OpenCV convention)
def _class_colour(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(cls_id * 1_000_003)
    r, g, b = rng.integers(80, 230, size=3)
    return int(b), int(g), int(r)


def _device_str(device_arg: str) -> str:
    """Convert CLI device arg to SAHI-compatible string."""
    if device_arg in ("", "auto"):
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg  # 'cpu', 'cuda:0', etc.


def _draw_overlay(image_path: Path, predictions: list[dict], out_path: Path) -> None:
    """Draw bounding boxes + labels on the full sheet and save as JPEG."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        colour = _class_colour(pred["cls_id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = f"{pred['cls_name']} {pred['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def predict(args: argparse.Namespace) -> None:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # ── resolve paths ────────────────────────────────────────────────────────
    if args.smoke:
        # use smoke weights + first available HF full-sheet image
        smoke_weights = Path("runs/detect/runs/detect/smoke/weights/best.pt")
        if not smoke_weights.exists():
            # fall back to any best.pt under runs/
            found = list(Path(".").glob("runs/**/best.pt"))
            if not found:
                raise FileNotFoundError(
                    "No smoke weights found. Run: python -m pidetect.detect.train --smoke"
                )
            smoke_weights = found[0]
        weights = smoke_weights

        hf_dir = Path("data/digitize-pid-yolo/DigitizePID_Dataset/images/train")
        sheets = sorted(hf_dir.glob("*.jpg"))
        if not sheets:
            raise FileNotFoundError(
                f"No full-sheet images in {hf_dir}. Run Phase 0 first."
            )
        image_path = sheets[0]
        device = "cpu"
        print(f"[smoke] weights : {weights}")
        print(f"[smoke] image   : {image_path}")
    else:
        if not args.weights:
            raise ValueError("--weights is required (or use --smoke)")
        if not args.image:
            raise ValueError("--image is required (or use --smoke)")
        weights = Path(args.weights)
        image_path = Path(args.image)
        device = _device_str(args.device)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = Path(args.out) if args.out else weights.parent.parent / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────────────
    print(f"\nLoading model: {weights}  (device={device})")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",          # SAHI wraps ultralytics; works for yolo11
        model_path=str(weights),
        confidence_threshold=args.conf,
        device=device,
    )

    # ── sliced inference ─────────────────────────────────────────────────────
    h_img = cv2.imread(str(image_path))
    if h_img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_h, img_w = h_img.shape[:2]
    print(f"Image size: {img_w}×{img_h}px")
    print(f"Tiling: {args.imgsz}px tiles, {args.overlap:.0%} overlap")

    t0 = time.perf_counter()
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
        slice_height=args.imgsz,
        slice_width=args.imgsz,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        perform_standard_pred=False,   # whole-image pass useless at 640px
        postprocess_type="GREEDYNMM",  # greedy NMM — best for dense seam regions
        postprocess_match_metric="IOS",
        postprocess_match_threshold=args.iou,
        auto_slice_resolution=False,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    # ── extract predictions ──────────────────────────────────────────────────
    predictions = []
    for obj in result.object_prediction_list:
        xyxy = obj.bbox.to_xyxy()
        predictions.append({
            "cls_id":   obj.category.id,
            "cls_name": obj.category.name,
            "conf":     round(float(obj.score.value), 4),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })

    # ── save JSON ────────────────────────────────────────────────────────────
    json_out = out_dir / "predictions.json"
    payload = {
        "image_path":    str(image_path),
        "image_size":    [img_w, img_h],
        "model":         str(weights),
        "tile_size":     args.imgsz,
        "overlap":       args.overlap,
        "conf_threshold": args.conf,
        "iou_threshold":  args.iou,
        "n_detections":  len(predictions),
        "inference_s":   round(elapsed, 2),
        "predictions":   predictions,
    }
    json_out.write_text(json.dumps(payload, indent=2))

    # ── save overlay ─────────────────────────────────────────────────────────
    overlay_out = out_dir / "overlay.jpg"
    _draw_overlay(image_path, predictions, overlay_out)

    # ── console summary ──────────────────────────────────────────────────────
    counts: Counter = Counter(p["cls_name"] for p in predictions)
    print(f"\n{'='*50}")
    print(f"Detections: {len(predictions)} total  ({elapsed:.1f}s)")
    print(f"{'='*50}")
    if counts:
        col_w = max(len(k) for k in counts) + 2
        for cls_name, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cls_name:<{col_w}} {n:>4}")
    else:
        print("  (no detections above confidence threshold)")
    print(f"{'='*50}")
    print(f"\nJSON    → {json_out}")
    print(f"Overlay → {overlay_out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIDetect — SAHI sliced full-sheet inference"
    )
    parser.add_argument("--weights", default=None,
                        help="Path to .pt weights (required unless --smoke)")
    parser.add_argument("--image", default=None,
                        help="Full-sheet image path (required unless --smoke)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: weights/../predict/)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tile size — must match training (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Tile overlap fraction (default: 0.2)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Per-tile confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Cross-tile NMM match threshold (default: 0.5)")
    parser.add_argument("--device", default="",
                        help="'' = auto, 'cpu', '0' = GPU 0")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke weights + first HF sheet on CPU")

    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
