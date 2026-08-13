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
- **Answers questions about the graph through a grounded LLM agent** — ask "how many valves have
  an MOV tag?" or "is sym_0 reliably connected to sym_33?" in a chat panel next to the overlay and
  get back an answer with the real tool calls behind it, not a plausible-sounding guess. The LLM
  never answers from its own P&ID knowledge; every claim is mechanically checked against the
  actual graph before it's returned.

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

### Grounded query agent — same honesty standard, including what's still open

A retrieval-only LLM agent sits on top of the graph (chat panel in the app), answering questions
through six narrow tools instead of general P&ID knowledge. It was adversarially stress-tested
with a 73-question fixture across three structurally different real sheets (dense/medium/sparse
density), specifically built to break it — and it did, five separate ways. Four are fixed
structurally (a mechanical check that rejects the answer, not a prompt asking nicely) and
confirmed live across **two different LLM providers**:

| Check | Result |
|---|---|
| Refuses questions needing general P&ID knowledge or a subjective verdict | **100%** — 3/3 sheets |
| Declines multi-hop "trace the pipe" questions rather than fabricating a route | **100%** — 3/3 sheets |
| Correctly reports absence for a real-but-missing subtype filter | **100%** — 3/3 sheets |
| Reports genuine "not found" for a compound multi-filter query | **75–100%** — 1 of 3 sheets still has a gap |

That last row is deliberately not rounded up: on two sheets, a specific compound-filter question
still hits its tool-call budget without answering, rather than declining cleanly — a known, named
gap, not a hidden one. One more open item: the guard that stops the agent from wastefully
re-verifying an answer it already has can, on certain multi-step questions, block a genuinely
necessary follow-up call. Full account of every gap found, what got fixed and how, and what's
still open: [docs/phase6_tier1_design.md](docs/phase6_tier1_design.md).

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
- **An agent that can't hallucinate past its own guardrails.** Every final answer is checked
  mechanically against the real tool-call transcript — not a second LLM call grading its own
  homework. Cited ids must trace to a real result, tag filters must trace to a real enumeration,
  and (after stress-testing caught the model rendering an unsupported verdict — "this sheet is
  well-instrumented" — on top of real numbers) a dedicated check now rejects evaluative
  conclusions the graph itself never licensed, even when every underlying fact was true. A failed
  check doesn't get flagged after the fact — it's rejected and the agent must retry or decline.
- **Provider-swappable by design.** The agent runs against Gemini, Groq, or a local Ollama model
  behind one interface, switched by a single config key — proved out for real when Gemini's
  free-tier daily quota exhausted mid-eval and the same test suite ran unmodified against Groq.

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
                     ──►  Chat panel ──► grounded LLM agent (6 tools) ──► mechanical grounding check
```

## Tech stack

**CV / ML:** PyTorch, YOLOv11 (Ultralytics), SAHI, PaddleOCR, OpenCV, scikit-image, skan
**Graph:** NetworkX
**Agent:** Gemini / Groq / Ollama behind a swappable provider interface, mechanical
(non-LLM) grounding checks
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

**To try the query agent's chat panel too**, export an API key before starting the backend
(`GEMINI_API_KEY` or `GROQ_API_KEY`, matching `configs/phase6.yaml`'s `provider` setting) — or
point it at a local Ollama model instead. Without a key, the chat panel shows a clear setup error
rather than crashing. No detection weights or upload needed to try it: the app pre-registers
"View Sheet 0 / 10 / 8 (Tier 1 agent demo)" buttons that jump straight to a completed OPEN100
result — the exact sheets the stress-test numbers above were measured on. Full setup:
[README_DEV.md](README_DEV.md).

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
