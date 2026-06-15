# PIDetect — P&ID Digitization

Detect ISA-5.1 instrument symbols, read their tags, trace pipelines, and extract the
process **connectivity graph** from P&ID diagrams. PyTorch · YOLOv11 · SAHI · OCR · graph extraction.

> Rework in progress. See `CLAUDE.md` for the full plan, architecture, and phase gates.

## Quickstart — rebuild dataset from scratch

```bash
python scripts/build_dataset.py
```

Runs the full Phase 0 pipeline (download → tile → synthetic → merge) with fixed seed 42.
Produces `data/merged/` ready for Phase 1 YOLOv11 training.
Skips steps whose outputs already exist; use `--force` to rebuild everything.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training on Colab/Kaggle

Full training runs on a free T4/P100/A100 GPU. The script handles dataset build,
Drive caching, training, and test-split evaluation in one go.

### Google Colab (recommended)

**Cell 1** — mount Drive once per session (dataset is cached there after the first build):
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2** — clone, build, train, evaluate:
```python
%cd /content
!git clone https://github.com/MalharRane/P-IDetect.git
%cd pidetect
!bash scripts/colab_setup.sh
```

First run builds the dataset (~15 min). Every subsequent session restores from Drive
in ~30 s. To force a full rebuild: `!bash scripts/colab_setup.sh --force-data`.

### Kaggle

Enable **Internet** in notebook Settings, then:
```bash
%%bash
git clone https://github.com/<your-username>/pidetect.git
cd pidetect
bash scripts/colab_setup.sh --no-drive
```

Dataset rebuilds each session (~15 min). To persist it, save the `data/` output
as a Kaggle dataset and symlink it on the next run.

### Batch size

Default is `--batch=16` (safe for T4/P100 16 GB). Override with e.g.
`!bash scripts/colab_setup.sh --batch=32` for a V100, or `--batch=64` for an A100.

### After training

The script runs evaluation automatically. To re-run manually (e.g. after downloading
`best.pt` to your laptop):
```bash
PYTHONPATH=src python -m pidetect.detect.evaluate \
  --weights runs/detect/train/weights/best.pt \
  --data    configs/yolo_baseline.yaml \
  --split   test
```

Outputs: overall mAP@50/50-95, per-class AP table (worst-first), confusion matrix PNG.

## Data
Public only. The Digitize-PID symbols set (`hamzas/digitize-pid-yolo`) is primary.
**NDA diagrams are never committed or trained on.**

## Phase 0 Progress

- [x] 0.1 — Environment & repo baseline (venv, imports verified, git + .gitignore, this checklist)
- [x] 0.2 — Download HF dataset (`hamzas/digitize-pid-yolo`) into `data/`
- [x] 0.3 — Dataset inspection: per-class counts, sample box overlays, tile-size driver
- [x] 0.4 — Tiling pipeline: 640px / 20% overlap → 35 826 tiles across train+val
- [x] 0.5 — Synthetic generator: 200 sheets, 32-class glyph library, YOLO labels + connectivity JSON
- [x] 0.6 — One-command build: `python scripts/build_dataset.py` → train/val/test split, histogram
