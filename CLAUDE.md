# CLAUDE.md — PIDetect

> Persistent project memory for Claude Code. Read this fully before proposing or writing anything.

## What this project is
PIDetect digitizes **P&IDs** (Piping & Instrumentation Diagrams): take a P&ID image →
output structured, machine-readable data — detected symbols, their tags, the pipelines,
and ultimately the **connectivity graph** (what connects to what). This is a from-scratch
rework of a hackathon project, aimed at a clean public repo, honest metrics, and a deployed demo.

## Current state (where we're starting from)
Rebuilding from scratch. The previous hackathon version was: YOLOv8-nano, ~18 training
images / ~127 boxes, an 8-class config that conflicted with a `nc:1` dataset yaml, no
rotation/scale handling, no tiling, and **detection only** (no OCR, no line detection, no
graph). Final metrics were precision 1.0 / recall 0.42 / mAP50 0.58 — classic tiny-data
overfit. We keep none of the model; we may reuse a React UI shell and a Flask skeleton later.

## Architecture (decided — don't relitigate without reason)
- **Symbol detection:** YOLOv11 (start `s`/`m`, not nano) + **SAHI tiling** for train & inference
  (full P&IDs are ~5000–7000px; never feed a whole sheet at 640px). Move to **YOLOv11-OBB**
  for rotated symbols once the axis-aligned baseline works.
- **Fine-grained classification:** two-stage. Detector finds a coarse symbol → a dedicated
  CNN classifies the crop (the check-valve family etc. are near-identical at tile scale).
- **Text/OCR:** PaddleOCR for tag detection + recognition; bind tags to symbols by geometry.
- **Lines:** classic CV — erase detected symbols+text, skeletonize, extract segments
  (LSD/Hough), classify solid vs dashed by gap analysis. Don't reach for a DL line model first.
- **Connectivity:** build a NetworkX graph (symbols + junctions = nodes; pipe segments = edges;
  crossings-without-a-dot = pass-over; off-page connectors = special nodes). Export JSON/GraphML.
- **Deploy:** FastAPI (tiled inference server-side) + React. Drop Streamlit.
- **Research stretch only:** Relationformer for joint detection+connection on PID2Graph.

## Data sources
- **Primary (symbols):** HF `hamzas/digitize-pid-yolo` — YOLO-format, 500 synthetic sheets,
  train/val 4:1. (It's the converted form of the Digitize-PID `.npy` Drive dump — use the HF
  version, don't wrestle the raw `.npy` files.)
- **Augment with:** Roboflow Universe sets, e.g. `pid-connect/p-id-symbols-r2`, `pid-dataset/p-id-ww1w9`.
- **Synthetic generator (we build this):** paste legend glyphs at random rotation/scale/position,
  draw process+signal lines, stamp tags → unlimited labels + covers rare classes. This is the
  data-scarcity unlock and the headline devlog story.
- **Connectivity eval:** PID2Graph (first public dataset with full graph ground truth).

## Hard rules
- **NDA:** the original hackathon diagrams are under NDA. NEVER commit them, never use them in
  training, never put them in screenshots/README. Public data only for anything publishable.
- **Never commit** `data/`, model weights (`*.pt`, `*.onnx`), or `runs/`. They're gitignored.
- **Code lives in `src/`, runs anywhere.** Training is launched on Colab/Kaggle GPU by cloning
  this repo and calling these modules. Notebooks are thin launchers ONLY — no logic in cells.
- **Honest metrics always:** report **per-class AP** (not just mAP — it hides rare-class failure),
  and for connectivity report **edge precision/recall**. No cherry-picked single numbers.

## Phase plan (each phase has a gate; don't skip ahead)
- **P0 Data:** download HF set, tiling prep, inspect (per-class counts + box viz), synth generator.
  Gate: regenerate dataset with one command; every class has enough instances.
- **P1 Detection:** YOLOv11 + SAHI + real aug (rotation on). Gate: recall ≥ ~0.85, per-class AP reported.
- **P2 Fine-grained:** two-stage classifier. Gate: confusion matrix on look-alike families.
- **P3 Text:** PaddleOCR + tag→symbol binding. Gate: tag exact-match + % symbols tagged.
- **P4 Lines+graph:** the differentiator. Gate: edge precision/recall vs PID2Graph / hand-labeled.
- **P5 Deploy:** FastAPI + React, upload→boxes+tags+graph+JSON.

## Conventions
- Python 3.10+, PyTorch, type hints, docstrings. Prefer small pure functions.
- Configs in `configs/*.yaml`, never hardcode paths — pass them in.
- Use Plan Mode for anything touching multiple files. Commit per working unit with clear messages.
- When unsure about a P&ID domain detail, ask rather than guess.
