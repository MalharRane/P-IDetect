# PIDetect — P&ID Digitization

Detect ISA-5.1 instrument symbols, read their tags, trace pipelines, and extract the
process **connectivity graph** from P&ID diagrams. PyTorch · YOLOv11 · SAHI · OCR · graph extraction.

> Rework in progress. See `CLAUDE.md` for the full plan, architecture, and phase gates.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 0 — get real numbers fast
```bash
python -m src.pidetect.data.download      # pull HF digitize-pid-yolo into data/
python -m src.pidetect.data.inspect       # per-class counts + sample box overlays
```

Training runs on Colab/Kaggle GPU — see `scripts/colab_setup.sh`.

## Data
Public only. The Digitize-PID symbols set (`hamzas/digitize-pid-yolo`) is primary.
**NDA diagrams are never committed or trained on.**

## Phase 0 Progress

- [x] 0.1 — Environment & repo baseline (venv, imports verified, git + .gitignore, this checklist)
- [ ] 0.2 — Download HF dataset (`hamzas/digitize-pid-yolo`) into `data/`
- [ ] 0.3 — Tiling prep: SAHI tile-size selection, overlap strategy, tile visualizer
- [ ] 0.4 — Dataset inspection: per-class counts, sample box overlays saved to `data/inspect/`
- [ ] 0.5 — Synthetic generator: paste glyphs, draw lines, stamp tags → labeled sheets
- [ ] 0.6 — Gate check: one-command dataset regeneration; every class has ≥ N instances
