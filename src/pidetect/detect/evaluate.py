"""Per-class AP evaluation harness.

Per CLAUDE.md: mAP alone is not acceptable reporting. Always print the full
per-class AP@50 table, sorted worst-first, so rare-class failures are visible.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


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


def _print_per_class_table(
    names: dict[int, str],
    ap_class_index: list[int],
    ap50_per_class: list[float],
    ap5095_per_class: list[float],
) -> None:
    """Print per-class AP table, worst AP@50 first."""
    rows = []
    for rank_idx, cls_idx in enumerate(ap_class_index):
        rows.append((names[cls_idx], ap50_per_class[rank_idx], ap5095_per_class[rank_idx]))

    # sort worst AP@50 first
    rows.sort(key=lambda r: r[1])

    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Class':<{col_w}}  {'AP@50':>8}  {'AP@50-95':>10}"
    print("\n" + "=" * len(header))
    print("Per-class AP  (worst first)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, ap50, ap5095 in rows:
        print(f"{name:<{col_w}}  {ap50:>8.3f}  {ap5095:>10.3f}")
    print("=" * len(header))


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
        _print_per_class_table(
            names=names,
            ap_class_index=list(box.ap_class_index),
            ap50_per_class=list(box.ap50),
            ap5095_per_class=list(box.ap),
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

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
