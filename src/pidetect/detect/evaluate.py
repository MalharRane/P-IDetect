"""Per-class AP evaluation harness.

Per CLAUDE.md: mAP alone is not acceptable reporting. Always print the full
per-class AP@50 table, sorted worst-first, so rare-class failures are visible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pidetect.data.open100 import (
    OUR_ARROW_IDX, OUR_INSTRUMENT_IDX, OUR_VALVE_IDX, SUPERCATEGORY_NAMES,
)

BASELINE_PATH = Path("docs/class_identity/phase1_baseline_ap.json")
AP_DROP_FLAG_THRESHOLD = 0.1


def _build_smoke_data(data_yaml: Path, weights: Path, n_images: int = 20) -> Path:
    """Mirror of train._build_smoke_data — build a tiny val-only subset for harness testing."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    # prefer test split; fall back to val
    val_key = "test" if "test" in cfg and (root / cfg["test"]).exists() else "val"
    val_img_dir = root / cfg[val_key]
    val_lbl_dir = val_img_dir.parent.parent / "labels" / Path(cfg[val_key]).name

    imgs = sorted(val_img_dir.glob("*"))[:n_images]
    if not imgs:
        raise FileNotFoundError(f"No images found in {val_img_dir}")

    tmp = Path(tempfile.mkdtemp(prefix="pidetect_eval_smoke_"))
    (tmp / "images" / "val").mkdir(parents=True)
    (tmp / "labels" / "val").mkdir(parents=True)

    for img in imgs:
        shutil.copy(img, tmp / "images" / "val" / img.name)
        lbl = val_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, tmp / "labels" / "val" / lbl.name)
        else:
            (tmp / "labels" / "val" / (img.stem + ".txt")).touch()

    smoke_cfg = dict(cfg)
    smoke_cfg["path"] = str(tmp)
    smoke_cfg["train"] = "images/val"  # Ultralytics requires train key; reuse val
    smoke_cfg["val"] = "images/val"
    smoke_cfg.pop("test", None)

    smoke_yaml = tmp / "smoke_eval.yaml"
    with open(smoke_yaml, "w") as f:
        yaml.dump(smoke_cfg, f)

    return smoke_yaml


def _load_baseline(path: Path = BASELINE_PATH) -> dict:
    """In-distribution per-class AP, transcribed from docs/phase1_analysis.md
    (subtask 1.6c) -- only 8 of 32 indices were ever individually recorded
    there. Missing indices report "not recorded" rather than a guess."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _print_per_class_table(
    names: dict[int, str],
    ap_class_index: list[int],
    ap50_per_class: list[float],
    ap5095_per_class: list[float],
    baseline: dict | None = None,
) -> None:
    """Print per-class AP table, worst AP@50 first. If `baseline` is given
    (see _load_baseline), also print the in-distribution AP@50 and the delta,
    flagging any class whose AP@50 dropped more than AP_DROP_FLAG_THRESHOLD."""
    by_idx = (baseline or {}).get("by_index", {})
    rows = []
    flagged = []
    for rank_idx, cls_idx in enumerate(ap_class_index):
        name = names[cls_idx]
        ap50 = ap50_per_class[rank_idx]
        ap5095 = ap5095_per_class[rank_idx]
        base = by_idx.get(str(cls_idx), {}).get("ap50") if baseline is not None else None
        delta = (ap50 - base) if base is not None else None
        if delta is not None and delta < -AP_DROP_FLAG_THRESHOLD:
            flagged.append((cls_idx, name, delta))
        rows.append((name, ap50, ap5095, base, delta))

    rows.sort(key=lambda r: r[1])

    col_w = max(len(r[0]) for r in rows) + 2
    if baseline is None:
        header = f"{'Class':<{col_w}}  {'AP@50':>8}  {'AP@50-95':>10}"
    else:
        header = (f"{'Class':<{col_w}}  {'AP@50':>8}  {'AP@50-95':>10}  "
                  f"{'In-dist AP50':>13}  {'Delta':>8}")
    print("\n" + "=" * len(header))
    print("Per-class AP  (worst first)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, ap50, ap5095, base, delta in rows:
        if baseline is None:
            print(f"{name:<{col_w}}  {ap50:>8.3f}  {ap5095:>10.3f}")
        else:
            base_s = f"{base:.3f}" if base is not None else "not recorded"
            delta_s = f"{delta:+.3f}" if delta is not None else "n/a"
            flag = "  <-- DROP > 0.1" if delta is not None and delta < -AP_DROP_FLAG_THRESHOLD else ""
            print(f"{name:<{col_w}}  {ap50:>8.3f}  {ap5095:>10.3f}  {base_s:>13}  {delta_s:>8}{flag}")
    print("=" * len(header))

    if baseline is not None:
        if flagged:
            print(f"\n[!] {len(flagged)} class(es) dropped >{AP_DROP_FLAG_THRESHOLD} AP@50 "
                  f"vs in-distribution -- next-iteration targets:")
            for cls_idx, name, delta in flagged:
                print(f"    idx {cls_idx:>2}  {name:<24}  {delta:+.3f}")
        else:
            recorded = sum(1 for r in rows if r[3] is not None)
            print(f"\n  No class with a recorded baseline dropped >{AP_DROP_FLAG_THRESHOLD} AP@50 "
                  f"({recorded}/{len(rows)} classes have a recorded in-distribution baseline).")


def evaluate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    split = args.split
    if args.smoke:
        print("[smoke] building tiny eval subset …")
        data_yaml = _build_smoke_data(data_yaml, weights, n_images=20)
        split = "val"
        print(f"[smoke] subset at {data_yaml.parent}")

    model = YOLO(str(weights))

    save_dir = Path(args.save_dir) if args.save_dir else weights.parent.parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=save_dir,
        plots=True,
    )

    box = metrics.box
    names: dict[int, str] = model.names  # {idx: class_name}

    print("\n" + "=" * 45)
    print("Overall")
    print("=" * 45)
    print(f"  mAP@50:      {box.map50:.4f}")
    print(f"  mAP@50-95:   {box.map:.4f}")
    print(f"  Precision:   {box.mp:.4f}")
    print(f"  Recall:      {box.mr:.4f}")
    print("=" * 45)

    if len(box.ap50) > 0:
        baseline = _load_baseline() if getattr(args, "realworld", False) else None
        _print_per_class_table(
            names=names,
            ap_class_index=list(box.ap_class_index),
            ap50_per_class=list(box.ap50),
            ap5095_per_class=list(box.ap),
            baseline=baseline,
        )
    else:
        print("\n[warn] No per-class AP data — check that the split has labels.")

    cm_path = save_dir / "confusion_matrix.png"
    if cm_path.exists():
        print(f"\nConfusion matrix → {cm_path}")
    else:
        # Ultralytics may name it differently; report what was saved
        saved = list(save_dir.glob("confusion_matrix*.png"))
        if saved:
            print(f"\nConfusion matrix → {saved[0]}")
        else:
            print("\n[warn] Confusion matrix image not found — Ultralytics may have skipped it (too few classes with detections).")

    print(f"\nAll outputs saved to {save_dir}\n")


# ---------------------------------------------------------------------------
# "open100" tier -- coarse 3-supercategory eval (subtask 1.6c)
#
# OPEN100's ground truth doesn't share our 32-class vocabulary (see
# src/pidetect/data/open100.py), so this can't go through model.val(). Only
# `_predict_tiles` touches ultralytics/torch -- everything below it is pure
# Python/numpy and was verified locally without a working model.
# ---------------------------------------------------------------------------

def _predict_tiles(model, images_dir: Path, conf: float, iou: float,
                    imgsz: int, device: str) -> dict[str, list[tuple]]:
    """Run inference on every tile. Returns {stem: [(cls, x1, y1, x2, y2, conf), ...]}
    in pixel coordinates (our 32-class indices, not yet remapped)."""
    preds: dict[str, list[tuple]] = {}
    for img_path in sorted(images_dir.glob("*.jpg")):
        results = model.predict(str(img_path), conf=conf, iou=iou, imgsz=imgsz,
                                device=device, verbose=False)
        boxes = []
        for b in results[0].boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            boxes.append((cls, x1, y1, x2, y2, float(b.conf.item())))
        preds[img_path.stem] = boxes
    return preds


def _remap_preds(
    preds: list[tuple], valve_idx: frozenset, arrow_idx: frozenset, instr_idx: frozenset,
) -> list[tuple]:
    """Collapse our 32-class predictions onto OPEN100's 3 supercategories
    (0=valve, 1=arrow, 2=instrument); drop predictions in classes OPEN100
    has no ground truth for at all (neither rewarded nor punished)."""
    out = []
    for cls, x1, y1, x2, y2, conf in preds:
        if cls in valve_idx:
            newc = 0
        elif cls in arrow_idx:
            newc = 1
        elif cls in instr_idx:
            newc = 2
        else:
            continue
        out.append((newc, x1, y1, x2, y2, conf))
    return out


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def _suppress_ignored(
    preds: list[tuple], ignore_boxes: list[tuple[float, float, float, float]],
    iou_thresh: float = 0.3,
) -> list[tuple]:
    """Drop predictions overlapping an OPEN100 'ignore' box (tank/pump/
    general/inlet-outlet -- real objects, just not in our taxonomy) instead
    of counting them as false positives. See SOURCES.md for the 'ignored,
    not wrong' rationale."""
    if not ignore_boxes:
        return preds
    kept = []
    for cls, x1, y1, x2, y2, conf in preds:
        if any(_iou((x1, y1, x2, y2), ig) >= iou_thresh for ig in ignore_boxes):
            continue
        kept.append((cls, x1, y1, x2, y2, conf))
    return kept


def _load_boxes_px(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        boxes.append((cls,
                      (xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                      (xc + bw / 2) * img_w, (yc + bh / 2) * img_h))
    return boxes


def _match(
    preds_by_image: dict[str, list[tuple]], gts_by_image: dict[str, list[tuple]],
    cls_id: int, iou_thresh: float,
) -> tuple[np.ndarray, int]:
    """Greedy highest-confidence-first IoU matching for one class across all
    images. Returns (tp: bool array sorted by descending confidence, n_gt)."""
    gts_only = {stem: [g[1:] for g in gts if g[0] == cls_id]
               for stem, gts in gts_by_image.items()}
    n_gt = sum(len(v) for v in gts_only.values())
    gt_used = {stem: [False] * len(v) for stem, v in gts_only.items()}

    all_preds = []
    for stem, preds in preds_by_image.items():
        for cls, x1, y1, x2, y2, conf in preds:
            if cls == cls_id:
                all_preds.append((conf, stem, (x1, y1, x2, y2)))
    all_preds.sort(key=lambda t: -t[0])

    tp = np.zeros(len(all_preds), dtype=bool)
    for i, (_, stem, box) in enumerate(all_preds):
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts_only.get(stem, [])):
            if gt_used[stem][j]:
                continue
            iou = _iou(box, gbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp[i] = True
            gt_used[stem][best_j] = True
    return tp, n_gt


def _ap_from_tp(tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP (area under the monotonic PR envelope),
    COCO-style. `tp` must already be sorted by descending confidence."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _ap50(preds_by_image: dict, gts_by_image: dict, cls_id: int) -> float:
    tp, n_gt = _match(preds_by_image, gts_by_image, cls_id, 0.5)
    return _ap_from_tp(tp, n_gt)


def _ap50_95(preds_by_image: dict, gts_by_image: dict, cls_id: int) -> float:
    aps = []
    for thr in np.arange(0.5, 1.0, 0.05):
        tp, n_gt = _match(preds_by_image, gts_by_image, cls_id, float(thr))
        aps.append(_ap_from_tp(tp, n_gt))
    return float(np.nanmean(aps))


def _baseline_for_indices(baseline: dict, indices: frozenset) -> tuple[float | None, int, int]:
    """Average whatever recorded AP@50 values fall within an OPEN100
    supercategory's our-index set. Returns (mean_ap50_or_None, n_recorded, n_total)."""
    by_idx = baseline.get("by_index", {})
    vals = [by_idx[str(i)]["ap50"] for i in sorted(indices)
            if str(i) in by_idx and by_idx[str(i)]["ap50"] is not None]
    mean = float(np.mean(vals)) if vals else None
    return mean, len(vals), len(indices)


def evaluate_open100(args: argparse.Namespace) -> None:
    """Score the model against the open100 tier's 3-supercategory ground
    truth (valve/arrow/instrument) -- see src/pidetect/data/open100.py for
    why this can't be a plain model.val() call."""
    from ultralytics import YOLO

    tier_dir = Path("data/realworld_eval/open100")
    images_dir = tier_dir / "images" / "test"
    labels_dir = tier_dir / "labels" / "test"
    ignore_dir = tier_dir / "ignore" / "test"
    if not images_dir.exists():
        print("[error] open100 tier not built yet. Run scripts/build_open100_eval.py first.")
        raise SystemExit(1)

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = YOLO(str(weights))

    raw_preds = _predict_tiles(model, images_dir, conf=args.conf, iou=args.iou,
                               imgsz=args.imgsz, device=args.device)

    preds_by_image: dict[str, list[tuple]] = {}
    gts_by_image: dict[str, list[tuple]] = {}
    from PIL import Image
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        with Image.open(img_path) as im:
            w, h = im.size
        remapped = _remap_preds(raw_preds.get(stem, []),
                                OUR_VALVE_IDX, OUR_ARROW_IDX, OUR_INSTRUMENT_IDX)
        ignore_boxes = [box[1:] for box in _load_boxes_px(ignore_dir / f"{stem}.txt", w, h)]
        preds_by_image[stem] = _suppress_ignored(remapped, ignore_boxes)
        gts_by_image[stem] = _load_boxes_px(labels_dir / f"{stem}.txt", w, h)

    baseline = _load_baseline()
    idx_sets = {0: OUR_VALVE_IDX, 1: OUR_ARROW_IDX, 2: OUR_INSTRUMENT_IDX}

    print("\n" + "=" * 70)
    print("open100 tier — per-supercategory AP  (coarse ground truth, see SOURCES.md)")
    print("=" * 70)
    header = (f"{'Supercategory':<14}  {'AP@50':>7}  {'AP@50-95':>9}  "
             f"{'In-dist AP50':>13}  {'Delta':>8}")
    print(header)
    print("-" * len(header))
    flagged = []
    for cls_id, name in SUPERCATEGORY_NAMES.items():
        ap50 = _ap50(preds_by_image, gts_by_image, cls_id)
        ap5095 = _ap50_95(preds_by_image, gts_by_image, cls_id)
        base, n_rec, n_tot = _baseline_for_indices(baseline, idx_sets[cls_id])
        if base is None:
            base_s, delta_s = "not recorded", "n/a"
        else:
            delta = ap50 - base
            base_s = f"{base:.3f} ({n_rec}/{n_tot} idx)"
            delta_s = f"{delta:+.3f}"
            if delta < -AP_DROP_FLAG_THRESHOLD:
                flagged.append((name, delta))
        print(f"{name:<14}  {ap50:>7.3f}  {ap5095:>9.3f}  {base_s:>13}  {delta_s:>8}")
    print("=" * 70)
    print(
        "\nNote: 'In-dist AP50' is the mean of whatever individual our-class indices in\n"
        "that supercategory were recorded in docs/phase1_analysis.md (often a partial\n"
        "sample, see n_rec/n_tot) -- not a full re-eval. Predictions on tank/pump/general/\n"
        "inlet-outlet objects were excluded from scoring entirely (ignored, not wrong)."
    )
    if flagged:
        print(f"\n[!] {len(flagged)} supercategory(ies) dropped >{AP_DROP_FLAG_THRESHOLD} AP@50 "
              f"vs in-distribution:")
        for name, delta in flagged:
            print(f"    {name:<14}  {delta:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PIDetect — per-class AP evaluation")

    parser.add_argument("--weights", required=True, help="Path to .pt weights file")
    parser.add_argument("--data", default="configs/yolo_baseline.yaml", help="Dataset YAML")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="'' = auto, 'cpu', '0' = GPU 0")
    parser.add_argument("--save-dir", dest="save_dir", default=None,
                        help="Where to write confusion matrix + plots (default: weights/../eval/)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run on 20-image subset to verify the harness — numbers are meaningless")
    parser.add_argument("--realworld", action="store_true",
                        help="Evaluate on a data/realworld_eval/<tier> set (out-of-distribution)")
    parser.add_argument("--tier", default="ood_synthetic",
                        choices=["ood_synthetic", "open100"],
                        help="Which real-world tier to use with --realworld "
                             "(default: ood_synthetic). See docs/realworld_eval_protocol.md")

    args = parser.parse_args()

    # --realworld overrides --data, resolved per-tier
    if args.realworld:
        args.data = f"data/realworld_eval/{args.tier}/{args.tier}.yaml"
        if not Path(args.data).exists():
            build_script = ("build_open100_eval.py" if args.tier == "open100"
                            else "build_ood_synthetic_eval.py")
            print(f"[error] '{args.tier}' tier not built yet.\n"
                  f"        Run: python scripts/{build_script}\n"
                  f"        See docs/realworld_eval_protocol.md for details.")
            raise SystemExit(1)

    if args.realworld and args.tier == "open100":
        evaluate_open100(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
