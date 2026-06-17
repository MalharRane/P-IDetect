"""
Subtask 1.7c — before/after comparison of two sets of YOLO weights.

Evaluates both the baseline model and a newly-trained model on:
  A) Tier-2 OPEN100 (3-supercategory AP: valve / arrow / instrument)
  B) In-distribution test split (overall 32-class mAP + flow_arrow per-class)

Writes docs/phase1_7_aug_results.md with a side-by-side table.

Usage (run locally after receiving new weights from Kaggle):
    PYTHONPATH=src python scripts/compare_weights.py \\
        --baseline runs/detect/train/weights/best.pt \\
        --new      runs/detect/train_small_objects/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

# ── reuse the low-level eval helpers from the main eval harness ───────────────
from pidetect.detect.evaluate import (
    _ap50, _ap50_95, _load_boxes_px, _match, _ap_from_tp,
    _remap_preds, _suppress_ignored,
)
from pidetect.data.open100 import (
    OUR_VALVE_IDX, OUR_ARROW_IDX, OUR_INSTRUMENT_IDX, SUPERCATEGORY_NAMES,
)

TIER2_IMAGES  = Path("data/realworld_eval/open100/images/test")
TIER2_LABELS  = Path("data/realworld_eval/open100/labels/test")
TIER2_IGNORE  = Path("data/realworld_eval/open100/ignore/test")
INDIST_CONFIG = "configs/yolo_baseline.yaml"
FLOW_ARROW_IDX = 23


# ── inference (copied from analyze_ood_failures.py) ──────────────────────────

def _predict_tiles(model, images_dir: Path, conf: float, iou: float,
                    imgsz: int, device: str) -> dict[str, list[tuple]]:
    preds: dict[str, list[tuple]] = {}
    for img_path in sorted(images_dir.glob("*.jpg")):
        results = model.predict(str(img_path), conf=conf, iou=iou,
                                imgsz=imgsz, device=device, verbose=False)
        boxes = []
        for b in results[0].boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            boxes.append((cls, x1, y1, x2, y2, float(b.conf.item())))
        preds[img_path.stem] = boxes
    return preds


# ── Tier-2 OPEN100 evaluation ─────────────────────────────────────────────────

def eval_open100(model, conf: float = 0.25, iou: float = 0.6,
                  imgsz: int = 640, device: str = "") -> dict:
    """Returns {supercategory_name: {ap50, ap5095, recall, n_gt, n_tp}}."""
    from PIL import Image

    if not TIER2_IMAGES.exists():
        raise FileNotFoundError(
            "OPEN100 eval tier not built. Run scripts/build_open100_eval.py first.")

    print("  Running Tier-2 inference …")
    raw_preds = _predict_tiles(model, TIER2_IMAGES, conf, iou, imgsz, device)

    preds_by_image: dict[str, list[tuple]] = {}
    gts_by_image:   dict[str, list[tuple]] = {}

    for img_path in sorted(TIER2_IMAGES.glob("*.jpg")):
        stem = img_path.stem
        with Image.open(img_path) as im:
            w, h = im.size
        remapped = _remap_preds(raw_preds.get(stem, []),
                                OUR_VALVE_IDX, OUR_ARROW_IDX, OUR_INSTRUMENT_IDX)
        ignore_boxes = [box[1:] for box in _load_boxes_px(TIER2_IGNORE / f"{stem}.txt", w, h)]
        preds_by_image[stem] = _suppress_ignored(remapped, ignore_boxes)
        gts_by_image[stem]   = _load_boxes_px(TIER2_LABELS / f"{stem}.txt", w, h)

    results = {}
    for cls_id, name in SUPERCATEGORY_NAMES.items():
        tp_arr, n_gt = _match(preds_by_image, gts_by_image, cls_id, 0.5)
        ap50   = _ap_from_tp(tp_arr, n_gt)
        ap5095 = _ap50_95(preds_by_image, gts_by_image, cls_id)
        n_tp   = int(tp_arr.sum()) if len(tp_arr) > 0 else 0
        recall = n_tp / n_gt if n_gt > 0 else float("nan")
        results[name] = {
            "ap50":   ap50,
            "ap5095": ap5095,
            "recall": recall,
            "n_gt":   n_gt,
            "n_tp":   n_tp,
        }
    return results


# ── in-distribution test split evaluation ────────────────────────────────────

def eval_indist(model, device: str = "") -> dict:
    """Returns {overall: {...}, flow_arrow: {...}}."""
    print("  Running in-dist model.val() …")
    metrics = model.val(
        data=INDIST_CONFIG,
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.6,
        device=device,
        plots=False,
        verbose=False,
    )
    box = metrics.box

    # per-class lookup
    cls_idx_list = list(map(int, box.ap_class_index))
    ap50_map  = dict(zip(cls_idx_list, map(float, box.ap50)))
    recall_map = dict(zip(cls_idx_list, map(float, box.r)))

    arrow_ap50   = ap50_map.get(FLOW_ARROW_IDX, float("nan"))
    arrow_recall = recall_map.get(FLOW_ARROW_IDX, float("nan"))

    return {
        "overall": {
            "map50":     float(box.map50),
            "map50_95":  float(box.map),
            "precision": float(box.mp),
            "recall":    float(box.mr),
        },
        "flow_arrow": {
            "ap50":   arrow_ap50,
            "recall": arrow_recall,
        },
    }


# ── markdown output ───────────────────────────────────────────────────────────

def _delta(a: float, b: float) -> str:
    if np.isnan(a) or np.isnan(b):
        return "n/a"
    d = b - a
    return f"{d:+.3f}"


def _fmt(v: float) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "n/a"


def write_results(
    baseline_open100: dict,
    new_open100: dict,
    baseline_indist: dict,
    new_indist: dict,
    baseline_label: str,
    new_label: str,
    out_path: Path,
    baseline_aug: str,
    new_aug: str,
) -> None:
    # Arrow recall detail (Tier-2): TP / n_gt; bucket counts not re-computed here
    arrow_b = baseline_open100["arrow"]
    arrow_n = new_open100["arrow"]

    # Build interpretation
    arrow_delta = new_open100["arrow"]["ap50"] - baseline_open100["arrow"]["ap50"]
    valve_delta = new_open100["valve"]["ap50"] - baseline_open100["valve"]["ap50"]
    indist_delta = new_indist["overall"]["map50"] - baseline_indist["overall"]["map50"]

    if arrow_delta > 0.05:
        arrow_interp = f"Arrow AP improved by {arrow_delta:+.3f} — scale jitter is working."
    elif arrow_delta > 0:
        arrow_interp = f"Arrow AP improved modestly ({arrow_delta:+.3f}). May need more epochs or higher scale."
    elif arrow_delta > -0.02:
        arrow_interp = f"Arrow AP essentially unchanged ({arrow_delta:+.3f}). Scale jitter not yet effective; check training curves."
    else:
        arrow_interp = f"Arrow AP regressed ({arrow_delta:+.3f}). Investigate: did copy_paste hurt precision?"

    if indist_delta < -0.01:
        indist_interp = f"In-dist mAP dropped {indist_delta:+.3f} — aggressive scale may be hurting in-distribution precision."
    elif indist_delta < 0:
        indist_interp = f"Tiny in-dist regression ({indist_delta:+.3f}), within noise."
    else:
        indist_interp = f"In-dist mAP held or improved ({indist_delta:+.3f}) — no regression."

    lines = [
        "# Phase 1.7c — Before/after: scale-focused augmentation",
        "",
        f"**Baseline:** `{baseline_label}` (aug profile: `{baseline_aug}`)  ",
        f"**New:**      `{new_label}` (aug profile: `{new_aug}`)  ",
        "**Eval date:** 2026-06-17  ",
        "",
        "Diagnosis driving this change: arrow misses are a **pure scale gap** — real OPEN100",
        "flow arrows are ~16 px (diagonal) vs ~79 px in synthetic training data (~5×). The",
        "`small_objects` profile uses `scale=0.9` to cover that range during training.",
        "",
        "---",
        "",
        "## A. Tier-2 OPEN100 — 3-supercategory AP@50",
        "",
        "*(real sheets, out-of-distribution; the honest generalization test)*",
        "",
        f"| Supercategory | {baseline_aug} AP@50 | {new_aug} AP@50 | Delta |",
        "|:--------------|-------------------:|----------------:|------:|",
    ]
    for name in ("valve", "arrow", "instrument"):
        b = baseline_open100[name]["ap50"]
        n = new_open100[name]["ap50"]
        key_marker = "  ← KEY" if name == "arrow" else ""
        lines.append(f"| {name:<13} | {_fmt(b):>18} | {_fmt(n):>15} | {_delta(b,n):>5} |{key_marker}")

    lines += [
        "",
        "### Arrow detail (Tier-2)",
        "",
        f"| Model          | n_gt | TP  | Recall | AP@50 |",
        "|:---------------|-----:|----:|-------:|------:|",
        f"| {baseline_aug:<14} | {arrow_b['n_gt']:>4} | {arrow_b['n_tp']:>3} | {_fmt(arrow_b['recall']):>6} | {_fmt(arrow_b['ap50']):>5} |",
        f"| {new_aug:<14} | {arrow_n['n_gt']:>4} | {arrow_n['n_tp']:>3} | {_fmt(arrow_n['recall']):>6} | {_fmt(arrow_n['ap50']):>5} |",
        "",
        "---",
        "",
        "## B. In-distribution test split (32-class, data/merged)",
        "",
        "*(synthetic test tiles; regression check — should stay ≥ 0.98)*",
        "",
        f"| Metric              | {baseline_aug} | {new_aug} | Delta |",
        "|:--------------------|----------:|------:|------:|",
        f"| mAP@50 (32-class)   | {_fmt(baseline_indist['overall']['map50']):>9} | {_fmt(new_indist['overall']['map50']):>5} | {_delta(baseline_indist['overall']['map50'], new_indist['overall']['map50']):>5} |",
        f"| mAP@50-95           | {_fmt(baseline_indist['overall']['map50_95']):>9} | {_fmt(new_indist['overall']['map50_95']):>5} | {_delta(baseline_indist['overall']['map50_95'], new_indist['overall']['map50_95']):>5} |",
        f"| flow_arrow AP@50    | {_fmt(baseline_indist['flow_arrow']['ap50']):>9} | {_fmt(new_indist['flow_arrow']['ap50']):>5} | {_delta(baseline_indist['flow_arrow']['ap50'], new_indist['flow_arrow']['ap50']):>5} |",
        f"| flow_arrow recall   | {_fmt(baseline_indist['flow_arrow']['recall']):>9} | {_fmt(new_indist['flow_arrow']['recall']):>5} | {_delta(baseline_indist['flow_arrow']['recall'], new_indist['flow_arrow']['recall']):>5} |",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"**Arrow (Tier-2):** {arrow_interp}  ",
        f"**Valve (Tier-2):** delta={_delta(baseline_open100['valve']['ap50'], new_open100['valve']['ap50'])} (valve fix requires more than aug alone — see valve root-cause).  ",
        f"**Regression check:** {indist_interp}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIDetect 1.7c — compare baseline vs new weights on OPEN100 + in-dist")
    parser.add_argument("--baseline", required=True, help="Baseline .pt weights")
    parser.add_argument("--new",      required=True, help="New .pt weights to compare")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.6)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="")
    parser.add_argument("--out", default="docs/phase1_7_aug_results.md")
    parser.add_argument("--baseline-aug", default="default",
                        help="Label for baseline aug profile (for table header)")
    parser.add_argument("--new-aug", default="small_objects",
                        help="Label for new aug profile (for table header)")
    args = parser.parse_args()

    from ultralytics import YOLO

    baseline_path = Path(args.baseline)
    new_path      = Path(args.new)
    for p in (baseline_path, new_path):
        if not p.exists():
            sys.exit(f"[error] weights not found: {p}")

    print(f"\n=== Baseline: {baseline_path} ===")
    baseline_model = YOLO(str(baseline_path))
    baseline_open100 = eval_open100(baseline_model, args.conf, args.iou, args.imgsz, args.device)
    baseline_indist  = eval_indist(baseline_model, args.device)

    print(f"\n=== New: {new_path} ===")
    new_model = YOLO(str(new_path))
    new_open100 = eval_open100(new_model, args.conf, args.iou, args.imgsz, args.device)
    new_indist  = eval_indist(new_model, args.device)

    # quick stdout summary
    print("\n--- OPEN100 delta ---")
    for name in ("valve", "arrow", "instrument"):
        b = baseline_open100[name]["ap50"]
        n = new_open100[name]["ap50"]
        print(f"  {name:<12}  {b:.3f} → {n:.3f}  ({_delta(b, n)})")
    print("--- In-dist mAP50 ---")
    print(f"  {baseline_indist['overall']['map50']:.4f} → {new_indist['overall']['map50']:.4f}  "
          f"({_delta(baseline_indist['overall']['map50'], new_indist['overall']['map50'])})")

    write_results(
        baseline_open100=baseline_open100,
        new_open100=new_open100,
        baseline_indist=baseline_indist,
        new_indist=new_indist,
        baseline_label=str(baseline_path),
        new_label=str(new_path),
        out_path=Path(args.out),
        baseline_aug=args.baseline_aug,
        new_aug=args.new_aug,
    )


if __name__ == "__main__":
    main()
