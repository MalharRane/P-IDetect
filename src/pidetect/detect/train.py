"""YOLOv11 training entrypoint — designed to be cloned and called on Colab/Kaggle GPU.

Smoke mode (--smoke) runs 1 epoch on 50 images on CPU to verify the pipeline without
needing a GPU or the full dataset.

Heavy rotation (degrees=180) helps real-world rotated symbols but can depress in-distribution
mAP on our axis-aligned synthetic test set — keep degrees moderate by default and treat
aggressive rotation as an ablation.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


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

    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        degrees=args.degrees,
        scale=args.scale,
        mosaic=args.mosaic,
        fliplr=args.fliplr,
        project="runs/detect",
        name="train" if not args.smoke else "smoke",
        exist_ok=args.smoke,
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
    parser.add_argument("--degrees", type=float, default=15.0,
                        help="Max rotation aug (°). Default 15 — keep moderate; 180 is an ablation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale jitter ± fraction")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")

    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on 50 images on CPU — just proves the pipeline runs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
