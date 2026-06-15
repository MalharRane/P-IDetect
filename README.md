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

Training runs on Colab/Kaggle GPU — see `scripts/colab_setup.sh`.

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
