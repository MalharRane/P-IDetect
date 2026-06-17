"""YOLOv11 training entrypoint — designed to be cloned and called on Colab/Kaggle GPU.

Smoke mode (--smoke) runs 1 epoch on 50 images on CPU to verify the pipeline without
needing a GPU or the full dataset.

Use --aug to select an augmentation profile:
  default       — original profile (degrees=15, scale=0.5, mosaic=1.0, fliplr=0.5)
  small_objects — scale-focused profile targeting the 5× arrow size gap diagnosed in
                  subtask 1.7a. scale=0.9 lets a 79 px synthetic arrow shrink to ~8 px,
                  covering the real-world ~16 px regime. degrees capped at 10 (orientation
                  is fine per diagnosis; rotation capacity is not needed).

When a non-default profile is active, individual --degrees/--scale/--mosaic/--fliplr flags
are ignored; the profile values take precedence.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Augmentation profiles — select with --aug
# ---------------------------------------------------------------------------

AUG_PROFILES: dict[str, dict] = {
    "default": {
        # Matches the original explicit CLI arg defaults — no behaviour change.
        "degrees": 15.0,
        "scale":   0.5,
        "mosaic":  1.0,
        "fliplr":  0.5,
    },
    "small_objects": {
        # Scale-focused profile (subtask 1.7c).
        # Diagnosis: arrows miss because real arrows are ~5× smaller than synthetic
        # (16 px vs 79 px median diagonal). scale=0.9 → zoom range [0.1×, 1.9×], so
        # a 79 px arrow can appear as ~8 px — well below the 16 px real-world median.
        # Orientation was confirmed NOT a factor; degrees kept low to avoid wasting
        # capacity on a dimension that doesn't need fixing.
        "degrees":    10.0,
        "scale":       0.9,
        "mosaic":      1.0,
        "fliplr":      0.5,
        "copy_paste":  0.1,   # extra small-object instances per batch
        "hsv_h":     0.015,   # mild colour jitter (ultralytics defaults)
        "hsv_s":       0.7,
        "hsv_v":       0.4,
    },
}


def _build_smoke_data(data_yaml: Path, n_images: int = 50) -> Path:
    """Copy up to n_images train images+labels into a temp dir and return a patched yaml."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    train_img_dir = root / cfg["train"]
    train_lbl_dir = train_img_dir.parent.parent / "labels" / "train"

    tmp = Path(tempfile.mkdtemp(prefix="pidetect_smoke_"))
    for split in ("train", "val"):
        (tmp / "images" / split).mkdir(parents=True)
        (tmp / "labels" / split).mkdir(parents=True)

    imgs = sorted(train_img_dir.glob("*"))[:n_images]
    if not imgs:
        raise FileNotFoundError(f"No training images found in {train_img_dir}")

    for img in imgs:
        shutil.copy(img, tmp / "images" / "train" / img.name)
        lbl = train_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, tmp / "labels" / "train" / lbl.name)
        else:
            (tmp / "labels" / "train" / (img.stem + ".txt")).touch()

    # val: reuse the same tiny set so validation doesn't crash
    for img in imgs[:max(1, n_images // 5)]:
        shutil.copy(img, tmp / "images" / "val" / img.name)
        lbl = train_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, tmp / "labels" / "val" / lbl.name)
        else:
            (tmp / "labels" / "val" / (img.stem + ".txt")).touch()

    smoke_cfg = dict(cfg)
    smoke_cfg["path"] = str(tmp)
    smoke_cfg["train"] = "images/train"
    smoke_cfg["val"] = "images/val"
    smoke_cfg.pop("test", None)

    smoke_yaml = tmp / "smoke.yaml"
    with open(smoke_yaml, "w") as f:
        yaml.dump(smoke_cfg, f)

    return smoke_yaml


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO  # import late so the module is importable without ultralytics

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    if args.smoke:
        print("[smoke] building tiny subset …")
        data_yaml = _build_smoke_data(data_yaml, n_images=50)
        epochs = 1
        device = "cpu"
        batch = 4
        print(f"[smoke] subset at {data_yaml.parent}")
    else:
        epochs = args.epochs
        device = args.device
        batch = args.batch

    model = YOLO(args.model)

    if args.aug == "default":
        aug_kwargs: dict = {
            "degrees": args.degrees,
            "scale":   args.scale,
            "mosaic":  args.mosaic,
            "fliplr":  args.fliplr,
        }
    else:
        aug_kwargs = AUG_PROFILES[args.aug]
        print(f"[aug] Profile '{args.aug}': {aug_kwargs}")

    run_name = f"train_{args.aug}" if args.aug != "default" else "train"
    if args.smoke:
        run_name = "smoke"

    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project="runs/detect",
        name=run_name,
        exist_ok=args.smoke,
        **aug_kwargs,
    )

    # DDP multi-GPU returns None from model.train() on worker processes
    if results is not None and getattr(results, "save_dir", None) is not None:
        save_dir = Path(results.save_dir)
        print(f"\nDone. Weights → {save_dir / 'weights'}")
        print(f"       Logs   → {save_dir}")
    else:
        weights = list(Path(".").glob("runs/**/weights/best.pt"))
        print(f"\nDone. Find weights at: {weights[0] if weights else 'runs/**/weights/best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PIDetect — YOLOv11 training harness")

    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics model or checkpoint")
    parser.add_argument("--data", default="configs/yolo_baseline.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Tile size (matches SAHI tiles)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="'' = auto, '0' = GPU 0, 'cpu' = CPU")

    # augmentation
    parser.add_argument(
        "--aug", default="default", choices=list(AUG_PROFILES),
        help="Augmentation profile (default | small_objects). When non-default, "
             "individual --degrees/--scale/--mosaic/--fliplr are ignored.",
    )
    parser.add_argument("--degrees", type=float, default=15.0,
                        help="Max rotation aug (°). Ignored when --aug is non-default.")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scale jitter ± fraction. Ignored when --aug is non-default.")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic probability. Ignored when --aug is non-default.")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Horizontal flip probability. Ignored when --aug is non-default.")

    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on 50 images on CPU — just proves the pipeline runs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
