"""Subtask 1.6c -- build the "ood_synthetic" real-world eval tier.

This is the existing single full-sheet SAHI demo check (the "SAMPLE Project"
sheet, `data/digitize-pid-yolo/.../images/train/0.jpg`), formalized and
labelled as a tier rather than left as an ad hoc demo run. Per the subtask:
"Keep, but label the tier" -- this does NOT try to fix its weakness, just
organizes it consistently with the new "open100" tier.

**Known weakness (disclosed, not fixed):** this sheet's tiles are in our
TRAIN split (`data/tiled/images/train/0_*.jpg`) -- the model has seen this
exact sheet during training. Evaluating it here re-tiles the full sheet fresh
(different window boundaries than the training tiles), but the visual
content is not held out. It is still useful as a full-sheet SAHI-style
sanity check and shares our exact 32-class vocabulary (unlike "open100",
which only supports coarse supercategories), but it is NOT a clean
out-of-distribution check -- see docs/realworld_eval_protocol.md.

Usage (from project root):
    python scripts/build_ood_synthetic_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.pidetect.data.tiling import slice_image

SRC_IMG = Path("data/digitize-pid-yolo/DigitizePID_Dataset/images/train/0.jpg")
SRC_LBL = Path("data/digitize-pid-yolo/DigitizePID_Dataset/labels/train/0.txt")
OUT_DIR = Path("data/realworld_eval/ood_synthetic")
BASELINE_YAML = Path("configs/yolo_baseline.yaml")


def main() -> None:
    if not SRC_IMG.exists():
        raise FileNotFoundError(
            f"{SRC_IMG} not found -- run scripts/build_dataset.py (step 1) first."
        )

    img_out = OUT_DIR / "images" / "test"
    lbl_out = OUT_DIR / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = slice_image(SRC_IMG, SRC_LBL, img_out, lbl_out, tile=640, overlap=0.2)
    print(f"  tiles written : {stats['tiles_written']}")
    print(f"  boxes in/out  : {stats['boxes_in']} / {stats['boxes_out']}")

    cfg = yaml.safe_load(BASELINE_YAML.read_text())
    yaml_text = (
        "# Built by scripts/build_ood_synthetic_eval.py -- subtask 1.6c.\n"
        "# Same 32-class verified taxonomy as configs/yolo_baseline.yaml --\n"
        "# this tier supports full per-index AP, unlike \"open100\".\n"
        "path: data/realworld_eval/ood_synthetic\n"
        "train: images/test   # required by Ultralytics even for test-only sets; reuse test\n"
        "val:   images/test\n"
        "test:  images/test\n\n"
        f"nc: {cfg['nc']}\n"
        "names:\n"
    )
    for i in range(cfg["nc"]):
        yaml_text += f"  {i}: {cfg['names'][i]}\n"
    (OUT_DIR / "ood_synthetic.yaml").write_text(yaml_text)
    print(f"  -> {OUT_DIR / 'ood_synthetic.yaml'}")

    print(
        "\n[!] KNOWN WEAKNESS: this sheet's tiles are in the TRAIN split "
        "(data/tiled/images/train/0_*.jpg). This is a re-tiled, same-content "
        "sanity check, NOT a held-out OOD check. See docs/realworld_eval_protocol.md."
    )


if __name__ == "__main__":
    main()
