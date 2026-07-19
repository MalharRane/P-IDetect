# PIDetect — P&ID Digitization

Detect ISA-5.1 instrument symbols, read their tags, trace pipelines, and extract the
process **connectivity graph** from P&ID diagrams. PyTorch · YOLOv11 · SAHI · OCR · graph extraction.

> Rework in progress. See `CLAUDE.md` for the full plan, architecture, and phase gates.

## Current state (2026-07-19)

**Detection (Phase 1) — strong.** YOLOv11s + SAHI. Held-out real-P&ID test split: mAP@50
0.994, mAP@50-95 0.985, recall 0.988. Cross-dataset generalization (OPEN100, fully independent
real-world sheets): valve CtrMt@50% **97.6%**, instrument CtrMt@50% **99.4%** — both clear the
80% gate. Full detail: `docs/phase1_analysis.md`, `docs/phase1_8_final.md`.

**Known-limited: `flow_arrow` detection.** OPEN100 arrow CtrMt@50% is **65.2%**, below the 80%
gate — a confirmed, diagnosed training-distribution gap (99.8% of real arrows are 8–30px;
synthetic training arrows have zero size overlap with that range, median 79.2px). Root cause is
training data, not model architecture; fix is adding real arrow crops as synth templates, not
a retrain. See `docs/arrow_triage/miss_breakdown.md`. This did not block Phase 2/4 — arrows are
tracked as a separate, lower-criticality item.

**Connectivity (Phase 4) — frozen, gate not met.** Three-sheet OPEN100 sample (sheets 0/3/10):
mean **P=0.434, R=0.759, F1=0.545** against a 0.70 F1 gate. **Gate not met.** The dominant
residual (32 of 58 total FPs, 55%) is a **shared-header pattern**: two symbols each tap
perpendicularly onto a common trunk pipe running parallel to the line between them, and the
corridor-ink test can't geometrically distinguish that from a dedicated point-to-point
connection. This residual has no surviving skeleton stub after erasure and no dashed-line
signal to key off (P3a = 0%), so it's unreachable by any discriminator built so far (ink,
stub-direction, line-type). Two upstream levers that could reach it — reducing erasure dilation
so stubs survive, and explicit shared-trunk/header detection — are named as future work, not
started. Full frozen record: `docs/phase4_final.md`.

**What works today in the connectivity pipeline:** symbol erasure → skeleton tracing → endpoint
binding → short-gap bridging → junction (connector/crossing) detection → contraction → GT-matched
edge scoring, all wired through one `configs/phase4.yaml`, with a config-wiring assertion
(`assert_run_step3_wired` in `src/pidetect/graph/lines.py`) that fails loudly if a documented
setting is ever silently dropped from a call site again (this happened once — see
`docs/phase4_final.md` §3.1 — and cost the project an extended period running a materially
different config than the one committed and documented).

**Other known-limited items:**
- `sym_106↔sym_110` (sheet 0): GT scores this instrument pair as connected with **zero visible
  ink** between them at any search margin — not a bug, there's nothing to detect.
- 4 remaining bbox-touch-no-ink false positives (twin instrument-tag pairs, overlapping icons)
  are indistinguishable from genuine connections by pixel count alone.
- 8 contraction over-bridge false positives (P3c): connector contraction transitively bridges
  symbols across chains of junction nodes (sometimes duplicate-split by the junction detector)
  that GT does not score as directly connected.
- Fine-grained valve/fitting sub-classification (Phase 2 scope) not yet started.

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
