# PIDetect

**Turning scanned P&ID drawings into structured, machine-readable data — symbols, tags, and the
process connectivity graph — with an end-to-end computer vision pipeline and a live web demo.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-frontend-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/data-public--only-lightgrey)](#data-sources)

![PIDetect: detected symbols, instrument tags, and connectivity overlay on a real P&ID sheet](docs/phase5_screenshots/hero_overlay_export.png)

## The problem

A P&ID (Piping & Instrumentation Diagram) is the master reference drawing for any process plant —
every valve, instrument, and pipe run a plant engineer needs is on it, but almost always locked in
a scanned image or PDF with no structured data behind it. Digitizing one by hand — finding every
symbol, transcribing every instrument tag, tracing every pipe run — is slow, manual, and repeated
across the industry on drawings that rarely change. **PIDetect automates the first pass**: upload a
sheet, get back detected symbols, OCR'd instrument tags, and a best-effort connectivity graph,
ready to inspect in the browser or hand off as JSON.

## What it does

- **Detects** valves, instruments, and fittings on full-resolution engineering drawings (sheets run
  1700–2800px+, well past a standard 640px detector's working range) using **YOLOv11 + SAHI
  tiling**, so nothing gets downsampled into invisibility.
- **Reads** every instrument tag (e.g. `TCV 1402`, `PT 14088`) with **PaddleOCR**, binding each
  reading to its symbol by geometry.
- **Traces pipe connectivity** with a classical CV pipeline — erase known symbols/text, skeletonize
  the remaining ink, extract line segments, detect junctions and crossings, and build a
  **NetworkX** graph of what connects to what.
- **Serves it as a real demo** — a FastAPI backend runs the full pipeline as an async job; a React
  frontend polls for progress, renders a pan/zoom overlay on the original sheet, and lets you
  click any detected symbol to inspect its class, confidence, bounding box, and parsed tag.
  Results export as JSON or an annotated PNG.

## Results — measured, not claimed

Every number below is scored against hand-verified ground truth on real-world sheets from the
public **PID2Graph OPEN100** dataset — not synthetic training data, and not cherry-picked. The same
"frozen, not a live per-upload score" numbers travel with every result the app produces, including
exported images.

| Stage | Metric | Result | Gate |
|---|---|---|---|
| Symbol detection | Node-match accuracy | **98.8%** | ≥ 80% — met |
| Instrument tag OCR | Exact-match (clean reads) | **90.7%** | ≥ 90% — met |
| Instrument tag OCR | Character error rate | **0.024** | — |
| Connectivity graph | F1 score | **0.549** | ≥ 0.70 — not yet met |

Detection and OCR are the strong, gate-passing parts of this pipeline. Connectivity — tracing
actual pipe runs from raw ink, the hardest and least standardized part of this problem — is the
acknowledged work-in-progress, shipped as a clearly-labeled "experimental" overlay layer rather
than hidden or inflated. **Two limitations were run to root cause instead of just reported:**
`flow_arrow` detection undershoots its gate (65.2%) because of a confirmed training-data scale
mismatch (real arrows are 8–30px, synthetic training arrows were 4x larger) — a labeled, scoped
fix, not a mystery. And 55% of connectivity's remaining false positives trace to one specific,
named geometric ambiguity (two symbols tapping the same shared trunk pipe) that's provably
unreachable by the discriminators built so far, with the two concrete upstream fixes that would
reach it written down as future work.

*Full methodology, per-sheet breakdowns, and every other measured limitation:
[README_DEV.md](README_DEV.md).*

## Engineering highlights

- **Gate-driven development.** Every phase (detection → OCR → connectivity → deploy) shipped
  against a pre-committed numeric gate, not a vibe check — including phases that didn't clear
  their gate, reported as such.
- **Full-resolution inference at scale.** SAHI tiling makes YOLOv11 usable on 2000px+ engineering
  drawings without the accuracy cliff of naive resizing.
- **Found and fixed a real detection-pipeline bug mid-project**: per-class NMS was letting the same
  physical valve survive as two duplicate graph nodes under different subtype guesses. Root-caused
  across 45 measured cross-class prediction pairs, fixed with a family-scoped dedup key, and backed
  with a construction-time regression assertion so it can't silently regress.
- **Classical CV where it's the right tool.** Line tracing uses skeletonization and morphology, not
  a second neural net — deliberately, since the geometry (thin, mostly-straight, solid-vs-dashed
  lines) doesn't need one.
- **A demo that's honest about its own cost.** Measured end-to-end latency (~2 minutes/sheet on
  CPU, OCR-dominated) drove a real architecture decision — async job + polling instead of a
  blocking request — documented with the actual numbers behind it, not an estimate.

## Architecture

```
P&ID sheet (image)
        │
        ▼
  YOLOv11 + SAHI tiling  ──────►  symbol boxes + classes
        │
        ▼
  Cross-subtype dedup (per-family NMS)
        │
        ▼
  PaddleOCR per instrument bubble  ──────►  instrument tags
        │
        ▼
  Erase symbols/text → skeletonize → trace line segments
        │
        ▼
  Junction / crossing detection → graph build + contraction
        │
        ▼
  NetworkX graph  ──────►  JSON / GraphML export
        │
        ▼
  FastAPI async job  ──►  React overlay (pan/zoom, click-to-inspect, PNG/JSON download)
```

## Tech stack

**CV / ML:** PyTorch, YOLOv11 (Ultralytics), SAHI, PaddleOCR, OpenCV, scikit-image, skan
**Graph:** NetworkX
**Backend:** FastAPI, Uvicorn
**Frontend:** React, Vite
**Data:** HuggingFace `hamzas/digitize-pid-yolo`, PID2Graph OPEN100, a custom synthetic P&ID
generator for training-data augmentation

## Try it locally

Requires **Python 3.12** (one interpreter runs detection and OCR together — see
[README_DEV.md](README_DEV.md) for why) and Node.js for the frontend.

```bash
git clone https://github.com/MalharRane/P-IDetect.git
cd P-IDetect

python3.12 -m venv .venv312
.venv312\Scripts\activate        # Windows; macOS/Linux: source .venv312/bin/activate
pip install -r requirements.txt
```

You'll need trained detection weights (gitignored — train your own via the Colab/Kaggle workflow
in [README_DEV.md](README_DEV.md), or point `configs/phase5.yaml`'s `weights_path` at one you have).

**Backend:**
```bash
PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False PYTHONPATH=src \
  uvicorn pidetect.api.main:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL, upload a P&ID sheet (e.g. one of
`data/realworld_eval/open100/_raw/*.png` after fetching OPEN100), and wait — it's slow (~2
min/sheet on CPU, see [README_DEV.md](README_DEV.md)) but real, end-to-end inference.

## Data sources

Public only, always. Primary training data is the HuggingFace `hamzas/digitize-pid-yolo` set;
evaluation is against PID2Graph OPEN100 real-world sheets. **The original hackathon project this
was rebuilt from used NDA'd diagrams — none of that data, or any model trained on it, appears
anywhere in this repo, its history, or its screenshots.**

## More detail

- [README_DEV.md](README_DEV.md) — full eval methodology, per-sheet metrics, every known
  limitation, and exact commands to reproduce every number above.
- [CLAUDE.md](CLAUDE.md) — architecture decisions, phase gates, and hard project rules.
- [docs/phase5_design.md](docs/phase5_design.md) — the deploy design doc, including a measured
  (not estimated) latency/hosting-cost analysis.
