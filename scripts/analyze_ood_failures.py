"""
Subtask 1.7a — OOD failure-mode diagnosis.

For every ground-truth valve and arrow box in the OPEN100 Tier-2 eval set,
classify the model outcome into exactly one bucket:

  TP           — correct supercategory prediction, IoU >= 0.5
  A NO_FIRE    — no prediction overlaps GT above a low IoU floor (< 0.1)
  B MISLOCATED — a prediction exists nearby but IoU < 0.5
  C WRONG_CLASS— well-localized (IoU >= 0.5) but wrong supercategory
  D EXCLUDED_IDX—well-localized, predicted index is in the excluded set
                  for this supercategory (measurement artifact, not failure)

Writes:
  docs/ood_failure_modes.md        breakdown tables (counts + %)
  docs/ood_examples/*.png          6-8 annotated tile images

Usage:
    PYTHONPATH=src python scripts/analyze_ood_failures.py \
        --weights runs/detect/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# ── supercategory constants ───────────────────────────────────────────────────
# mirrors src/pidetect/data/open100.py but extended with excluded sets
VALVE_CORRECT_IDX  = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
VALVE_EXCLUDED_IDX = frozenset({14, 15, 16, 17, 18, 24})
# 14/15/24 = ambiguous per mapping.md; 16/17/18 = spectacle-blind, not valves

ARROW_CORRECT_IDX  = frozenset({23})
ARROW_EXCLUDED_IDX: frozenset[int] = frozenset()   # no excluded arrow indices

# GT supercategory class ids (from labels/test/*.txt)
GT_VALVE_CLS = 0
GT_ARROW_CLS = 1

LOW_IOU_FLOOR = 0.1   # below this → NO_FIRE; at or above → at least MISLOCATED

BUCKETS = ("TP", "A", "B", "C", "D")
BUCKET_LABELS = {
    "TP": "TP (matched)",
    "A":  "A — NO_FIRE",
    "B":  "B — MISLOCATED",
    "C":  "C — WRONG_CLASS",
    "D":  "D — EXCLUDED_IDX (artifact)",
}


# ── geometry helpers (mirrors evaluate.py private helpers) ────────────────────

def _iou(a: tuple[float, float, float, float],
          b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max((ax2 - ax1) * (ay2 - ay1), 1e-9)
    b_area = max((bx2 - bx1) * (by2 - by1), 1e-9)
    return inter / (a_area + b_area - inter)


def _load_boxes_px(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Load YOLO-format label file → list of (cls, x1, y1, x2, y2) in pixels."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        boxes.append((cls,
                      (xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                      (xc + bw / 2) * img_w, (yc + bh / 2) * img_h))
    return boxes


# ── categorization ────────────────────────────────────────────────────────────

def categorize(
    gt_box: tuple,
    raw_preds: list[tuple],
    supercategory: str,
) -> tuple[str, dict]:
    """Assign one bucket to a single GT box.

    gt_box  : (cls, x1, y1, x2, y2) in pixels (cls not used here)
    raw_preds: [(cls, x1, y1, x2, y2, conf), ...] raw 32-class predictions

    Returns (bucket, details) where bucket in ("TP","A","B","C","D") and
    details carries the best prediction and IoU for the annotator.
    """
    gt_coords = gt_box[1:]  # (x1, y1, x2, y2)

    if supercategory == "valve":
        correct_idx  = VALVE_CORRECT_IDX
        excluded_idx = VALVE_EXCLUDED_IDX
    else:
        correct_idx  = ARROW_CORRECT_IDX
        excluded_idx = ARROW_EXCLUDED_IDX

    # Pass 1: is there a TP? (correct class, IoU >= 0.5)
    for pred in raw_preds:
        pcls, px1, py1, px2, py2, pconf = pred
        if pcls in correct_idx and _iou(gt_coords, (px1, py1, px2, py2)) >= 0.5:
            return "TP", {"iou": _iou(gt_coords, (px1, py1, px2, py2)), "pred": pred}

    # Pass 2: best prediction by IoU (any class)
    best_iou = 0.0
    best_pred: tuple | None = None
    for pred in raw_preds:
        _, px1, py1, px2, py2, _ = pred
        iou = _iou(gt_coords, (px1, py1, px2, py2))
        if iou > best_iou:
            best_iou = iou
            best_pred = pred

    if best_iou < LOW_IOU_FLOOR:
        return "A", {"best_iou": best_iou, "best_pred": best_pred}
    if best_iou < 0.5:
        return "B", {"best_iou": best_iou, "best_pred": best_pred}

    # best_iou >= 0.5
    assert best_pred is not None
    pcls = best_pred[0]
    if pcls in excluded_idx:
        return "D", {"iou": best_iou, "pred": best_pred}
    return "C", {"iou": best_iou, "pred": best_pred}


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, images_dir: Path, conf: float, iou: float,
                   imgsz: int, device: str) -> dict[str, list[tuple]]:
    """Raw 32-class predictions per tile stem (pixel coords)."""
    preds: dict[str, list[tuple]] = {}
    img_paths = sorted(images_dir.glob("*.jpg"))
    print(f"  Running inference on {len(img_paths)} tiles …")
    for img_path in img_paths:
        results = model.predict(str(img_path), conf=conf, iou=iou, imgsz=imgsz,
                                device=device, verbose=False)
        boxes = []
        for b in results[0].boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            boxes.append((cls, x1, y1, x2, y2, float(b.conf.item())))
        preds[img_path.stem] = boxes
    return preds


# ── visualization ─────────────────────────────────────────────────────────────

def _draw_box_cv2(img, box: tuple[float, float, float, float],
                   color: tuple[int, int, int], label: str = "") -> None:
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_tile(
    img_path: Path,
    gt_box: tuple,
    details: dict,
    bucket: str,
    supercategory: str,
    model_names: dict[int, str],
    out_path: Path,
) -> None:
    """Draw GT (green) + best prediction (red) on a copy of the tile and save."""
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError("cv2 returned None")

        gt_coords = gt_box[1:]
        _draw_box_cv2(img, gt_coords, (0, 200, 0),
                      label=f"GT:{supercategory}")

        pred = details.get("pred") or details.get("best_pred")
        iou  = details.get("iou") or details.get("best_iou", 0.0)
        if pred is not None:
            pcls = pred[0]
            pname = model_names.get(pcls, f"idx{pcls}")
            _draw_box_cv2(img, pred[1:5], (0, 60, 220),
                          label=f"idx{pcls}:{pname[:10]}  IoU={iou:.2f}")
        else:
            h, w = img.shape[:2]
            import cv2 as _cv2
            _cv2.putText(img, "NO PREDICTION", (10, 20),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 220), 2)

        # bucket banner across top
        banner = f"[{bucket}] {BUCKET_LABELS[bucket]}  |  {supercategory}"
        cv2.putText(img, banner, (6, img.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return

    except Exception:
        pass  # fall through to PIL fallback

    # PIL fallback
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    gt_coords = gt_box[1:]
    draw.rectangle(list(gt_coords), outline=(0, 200, 0), width=2)
    draw.text((gt_coords[0], gt_coords[1] - 12), f"GT:{supercategory}",
              fill=(0, 200, 0))

    pred = details.get("pred") or details.get("best_pred")
    iou  = details.get("iou") or details.get("best_iou", 0.0)
    if pred is not None:
        pcls = pred[0]
        pname = model_names.get(pcls, f"idx{pcls}")
        draw.rectangle(list(pred[1:5]), outline=(220, 60, 0), width=2)
        draw.text((pred[1], pred[2] - 12),
                  f"idx{pcls}:{pname[:10]} IoU={iou:.2f}", fill=(220, 60, 0))
    else:
        draw.text((10, 10), "NO PREDICTION", fill=(220, 60, 0))

    draw.text((6, pil_img.height - 18),
              f"[{bucket}] {BUCKET_LABELS[bucket]}  |  {supercategory}",
              fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path)


# ── example selection ─────────────────────────────────────────────────────────

def select_examples(
    all_results: list[dict],
    n_target: int = 8,
) -> list[dict]:
    """Pick up to n_target results with diversity across buckets.

    Strategy: 1 from each existing (supercategory, bucket) pair in failure-bucket
    order (A-D, then TP), then fill remaining slots with a second pick per pair
    in the same order — so every failure bucket appears before any bucket gets 3+.
    """
    slots: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["supercategory"], r["bucket"])
        slots[key].append(r)

    # failure buckets first (D is the artifact bucket we specifically want to show), then TP
    order = [(cat, bkt) for bkt in ("A", "B", "C", "D", "TP")
             for cat in ("valve", "arrow")]

    # Two passes: first pick 1 per slot, then pick a second from any slot with extras
    selected: list[dict] = []
    for pass_offset in (0, 1):
        for key in order:
            pool = slots.get(key, [])
            if len(pool) > pass_offset:
                selected.append(pool[pass_offset])
                if len(selected) >= n_target:
                    return selected
    return selected


# ── markdown output ───────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def write_markdown(
    valve_counts: dict[str, int],
    arrow_counts: dict[str, int],
    wrong_class_preds: dict[str, list[int]],
    n_examples: int,
    out_path: Path,
    model_names: dict[int, str],
) -> None:
    def table_rows(counts: dict[str, int]) -> str:
        total = sum(counts.values())
        rows = []
        for bkt in BUCKETS:
            n = counts.get(bkt, 0)
            rows.append(f"| {BUCKET_LABELS[bkt]:<36} | {n:>5} | {_pct(n, total):>8} |")
        rows.append(f"| {'**Total**':<36} | {total:>5} | {'100%':>8} |")
        return "\n".join(rows)

    def artifact_vs_real(counts: dict[str, int]) -> str:
        total_fail = sum(counts.get(b, 0) for b in ("A", "B", "C", "D"))
        d = counts.get("D", 0)
        abc = total_fail - d
        if total_fail == 0:
            return "  No failures recorded."
        return (f"  D (artifact):   {d:>4} / {total_fail}  ({_pct(d, total_fail)})\n"
                f"  A+B+C (real):   {abc:>4} / {total_fail}  ({_pct(abc, total_fail)})")

    # wrong-class predicted indices summary
    def wc_summary(cat: str) -> str:
        preds = wrong_class_preds.get(cat, [])
        if not preds:
            return "  (none)"
        from collections import Counter
        counts_c = Counter(preds)
        lines = []
        for idx, n in counts_c.most_common(8):
            lines.append(f"  idx {idx:>2}  {model_names.get(idx, '?'):<28}  × {n}")
        return "\n".join(lines)

    lines = [
        "# OOD Failure-Mode Diagnosis (subtask 1.7a)",
        "",
        f"**Model:** `runs/detect/train/weights/best.pt`  ",
        "**Eval set:** OPEN100 Tier 2 (12 real sheets, 197 tiles)  ",
        "**Date:** 2026-06-17  ",
        "**Inference conf threshold:** 0.01 (low, to avoid masking B-bucket cases)  ",
        "",
        "Bucket definitions:",
        "- **TP** — correct supercategory prediction, IoU ≥ 0.5",
        "- **A — NO_FIRE** — no prediction overlaps GT above IoU 0.1 (true miss)",
        "- **B — MISLOCATED** — prediction exists nearby but IoU < 0.5",
        "- **C — WRONG_CLASS** — well-localized (IoU ≥ 0.5) but different supercategory",
        "- **D — EXCLUDED_IDX** — well-localized, predicted index is one we excluded from",
        "  the supercategory mapping (14/15/16/17/18/24 for valve). **Measurement artifact.**",
        "",
        "---",
        "",
        "## Valve failure breakdown",
        "",
        "| Bucket                               | Count |      % |",
        "|--------------------------------------|------:|-------:|",
        table_rows(valve_counts),
        "",
        "**Artifact vs real (non-TP only):**",
        artifact_vs_real(valve_counts),
        "",
        "Wrong-class predicted indices (bucket C, valve GT):",
        wc_summary("valve"),
        "",
        "---",
        "",
        "## Arrow failure breakdown",
        "",
        "| Bucket                               | Count |      % |",
        "|--------------------------------------|------:|-------:|",
        table_rows(arrow_counts),
        "",
        "**Artifact vs real (non-TP only):**",
        artifact_vs_real(arrow_counts),
        "",
        "Wrong-class predicted indices (bucket C, arrow GT):",
        wc_summary("arrow"),
        "",
        "---",
        "",
        "## Key question",
        "",
        "**How much of the valve AP drop is bucket D (artifact) vs A/B/C (real)?**",
        "",
        artifact_vs_real(valve_counts),
        "",
        "Interpretation: bucket D cases are where the model found something at the right",
        "location but our supercategory mapping deliberately excluded the predicted index",
        "(spectacle-blind family 16/17/18 and ambiguous circles 14/15/24). These count as",
        "misses in AP scoring even though the model 'saw' something there. A/B/C cases are",
        "genuine generalization failures — the model either missed the symbol entirely (A),",
        "roughly localized it but not well enough (B), or confused it with a different class (C).",
        "",
        "---",
        "",
        "## Annotated examples",
        "",
        f"See `docs/ood_examples/` ({n_examples} tiles).  ",
        "Green box = GT. Red/orange box = best prediction. Bucket label in lower-left corner.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Written: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIDetect subtask 1.7a — OOD failure-mode diagnosis")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--conf",   type=float, default=0.01,
                        help="Inference confidence threshold (low to catch B cases)")
    parser.add_argument("--iou",    type=float, default=0.6)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[error] weights not found: {weights}")

    tier_dir   = Path("data/realworld_eval/open100")
    images_dir = tier_dir / "images" / "test"
    labels_dir = tier_dir / "labels" / "test"
    ignore_dir = tier_dir / "ignore"  / "test"
    if not images_dir.exists():
        sys.exit("[error] open100 tier not built. Run scripts/build_open100_eval.py first.")

    examples_dir = Path("docs/ood_examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    sys.path.insert(0, "src")
    from ultralytics import YOLO
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    model_names: dict[int, str] = model.names

    # ── inference ─────────────────────────────────────────────────────────────
    raw_preds = run_inference(model, images_dir, conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, device=args.device)

    # ── per-GT categorization ─────────────────────────────────────────────────
    from PIL import Image

    valve_counts: dict[str, int] = defaultdict(int)
    arrow_counts: dict[str, int] = defaultdict(int)
    wrong_class_preds: dict[str, list[int]] = {"valve": [], "arrow": []}
    all_results: list[dict] = []

    img_paths = sorted(images_dir.glob("*.jpg"))
    for img_path in img_paths:
        stem = img_path.stem
        with Image.open(img_path) as im:
            w, h = im.size

        gt_boxes    = _load_boxes_px(labels_dir / f"{stem}.txt", w, h)
        ignore_boxes = _load_boxes_px(ignore_dir / f"{stem}.txt", w, h)
        raw         = raw_preds.get(stem, [])

        for gt_box in gt_boxes:
            gt_cls = gt_box[0]
            if gt_cls == GT_VALVE_CLS:
                supercategory = "valve"
                counts = valve_counts
            elif gt_cls == GT_ARROW_CLS:
                supercategory = "arrow"
                counts = arrow_counts
            else:
                continue  # instrument — not analysed here

            # Skip GT boxes that overlap an ignore box (shouldn't happen,
            # but the ignore tiles are separate; just a safety guard)
            gt_coords = gt_box[1:]
            if any(_iou(gt_coords, ib[1:]) >= 0.5 for ib in ignore_boxes):
                continue

            bucket, details = categorize(gt_box, raw, supercategory)
            counts[bucket] += 1

            if bucket == "C":
                pred = details.get("pred") or details.get("best_pred")
                if pred is not None:
                    wrong_class_preds[supercategory].append(pred[0])

            all_results.append({
                "supercategory": supercategory,
                "bucket":        bucket,
                "details":       details,
                "img_path":      img_path,
                "gt_box":        gt_box,
            })

    # ── print summary ─────────────────────────────────────────────────────────
    for cat, counts in (("valve", valve_counts), ("arrow", arrow_counts)):
        total = sum(counts.values())
        print(f"\n{cat.upper()} ({total} GT boxes):")
        for bkt in BUCKETS:
            n = counts.get(bkt, 0)
            print(f"  {BUCKET_LABELS[bkt]:<36}  {n:>4}  ({_pct(n, total)})")

    # ── annotated examples ────────────────────────────────────────────────────
    examples = select_examples(all_results, n_target=args.n_examples)
    counters: dict[tuple[str, str], int] = defaultdict(int)
    saved_examples: list[Path] = []
    for r in examples:
        cat = r["supercategory"]
        bkt = r["bucket"]
        counters[(cat, bkt)] += 1
        n = counters[(cat, bkt)]
        out_name = f"{cat}_bucket_{bkt}_{n:03d}.png"
        out_path = examples_dir / out_name
        annotate_tile(
            img_path=r["img_path"],
            gt_box=r["gt_box"],
            details=r["details"],
            bucket=bkt,
            supercategory=cat,
            model_names=model_names,
            out_path=out_path,
        )
        saved_examples.append(out_path)
        print(f"  example → {out_path.name}")

    # ── write markdown ────────────────────────────────────────────────────────
    write_markdown(
        valve_counts=dict(valve_counts),
        arrow_counts=dict(arrow_counts),
        wrong_class_preds=wrong_class_preds,
        n_examples=len(saved_examples),
        out_path=Path("docs/ood_failure_modes.md"),
        model_names=model_names,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
