# PIDetect — Engineering Log (Dev README)

> This is the full technical record: phase gates, exact eval numbers, known residuals, and
> the commands to reproduce every result below. For the short, recruiter-facing overview,
> see [README.md](README.md). For the persistent project plan/architecture decisions Claude
> Code reads before touching this repo, see [CLAUDE.md](CLAUDE.md).

Upload a P&ID (Piping & Instrumentation Diagram) sheet → get back detected instrument/valve
symbols, their instrument tags (read via OCR), and a best-effort process **connectivity graph** —
viewable in the browser as an overlay on the original sheet, and downloadable as JSON or as an
annotated PNG. A grounded LLM agent (chat panel, Phase 6) answers questions about that graph
through mechanically-checked tool calls rather than general knowledge. PyTorch · YOLOv11 · SAHI ·
PaddleOCR · NetworkX · Gemini/Groq/Ollama.

![PIDetect overlay on a real OPEN100 sheet, with legend and honest-metrics footer](docs/phase5_screenshots/hero_overlay_export.png)

> Rework in progress. See `CLAUDE.md` for the full plan, architecture, and phase gates.
> This README covers the shipped v1 demo (Phase 5); `docs/phase4_final.md` and
> `docs/phase3_results.md` are the frozen evaluation records those numbers below are drawn from.

## Demo video

*(placeholder — a short screen recording of the golden path: upload → wait → overlay with tags
and graph → JSON/PNG download, on a real OPEN100 sheet, goes here.)*

## Honest metrics

No cherry-picked single number — this is what actually measures on the 3-sheet OPEN100 held-out
sample (`docs/phase4_final.md`, `docs/phase3_results.md`), reported the same way in the app's own
UI (a "frozen, not a live per-upload score" caption travels with every result, including the
exported PNG).

| Stage | Metric | Result | Gate |
|---|---|---|---|
| Symbol detection | Node-match (CtrMt@50%) | **98.8%** | ≥ 80% — met |
| Instrument tag OCR | Exact-match (ok/ok_placeholder rows) | **90.7%** | ≥ 90% — met |
| Instrument tag OCR | Micro-averaged CER | **0.0241** | — |
| Connectivity graph | F1 (precision 0.435, recall 0.775) | **0.549** | ≥ 0.70 — **not met** |

Connectivity is explicitly labeled **experimental** in the UI (a secondary, lighter, toggleable
overlay layer, never styled with the same visual confidence as a detection box) — it's below its
own gate and should not be read as authoritative. Detection and OCR are the strong, gate-passing
parts of this pipeline; connectivity is the acknowledged differentiator-in-progress.

## Known limitations

- **`flow_arrow` detection: 65.2%** node-match on OPEN100, below the 80% gate. Confirmed root
  cause: synthetic training arrows have zero size overlap with real arrows (99.8% of real arrows
  are 8–30px; synthetic ones median 79.2px). This is a training-data gap, not an architecture
  problem — fix is adding real arrow crops as synth templates, not a retrain. Did not block later
  phases; tracked separately. See `docs/arrow_triage/miss_breakdown.md`.
- **Connectivity's dominant residual: shared-header false positives** — 32 of 58 total FPs (55%)
  are two symbols each tapping perpendicularly onto a common trunk pipe, which the corridor-ink
  test can't geometrically distinguish from a dedicated point-to-point connection. No surviving
  skeleton stub after erasure, no dashed-line signal to key off — unreachable by any discriminator
  built so far. Two upstream levers (reduced erasure dilation; explicit shared-trunk detection)
  are named as future work, not started. Full detail: `docs/phase4_final.md` §5.
- **10 open cross-node-type close-prediction pairs** (`docs/phase4_final.md` §2) — valve/instrument
  detections sitting very close to an `unknown_fitting`/`flow_arrow`/`tag_rect` detection. Each
  needs individual visual judgment (a valve actuator can legitimately sit right next to an
  instrument bubble as two real objects) — deliberately NOT resolved by a blanket rule.
  Not investigated this pass.
- **~2 minutes per sheet on CPU** (measured end-to-end, OPEN100-sized sheets ~1700–2800px):
  detection ~10–20s, OCR ~75–95s (dominant cost — a spike into batching/process-pooling found no
  win, since PaddleOCR's CPU inference already saturates available compute per call; one real
  optimization — disabling unneeded doc-orientation/unwarping sub-models — gave a real but modest
  ~24% OCR reduction, already applied), graph construction ~10–20s. This rules out a synchronous
  request/response UI; the app uses an async job with polling instead. Real production sheets
  (CLAUDE.md: 5000–7000px) are untested and would likely run substantially longer.
- **81.7% of rendered connectivity edges are inferred straight chords, not traced pipe routes**
  (measured on a live sheet: 116 of 142 edges have no traceable skeleton polyline — short-gap
  ink-inferred connections and contracted-through connector/crossing edges alike). The UI marks
  these visually distinct (dotted, fainter) from the 18.3% that do have a real traced route, with
  an explicit legend entry — but it's worth knowing the split going in.
- Vessels (tanks/pumps) are not a YOLO-detected class in this pipeline yet — they appear in the
  OPEN100 evaluation only via ground-truth annotation, not in a real upload's output.
- Fine-grained valve/fitting sub-classification (Phase 2 scope) not yet started.
- Single in-memory job store, one worker — concurrent uploads serialize. Fine for a demo, not for
  production (`docs/phase5_design.md`).

## Phase 6 — Grounded Query Agent

A retrieval-only LLM agent answers questions about a completed job's graph through six narrow
tools (`find_nodes`, `get_node`, `get_neighbors`, `count_nodes`, `list_systems_or_lines`,
`resolve_term`) — never from its own P&ID/ISA knowledge. Full design, including every gap found
and fixed: `docs/phase6_tier1_design.md`.

### Grounding architecture

The core discipline: the LLM only orchestrates tool calls; every factual claim in a final answer
is checked mechanically against the tool-call transcript before it's returned — not judged by a
second LLM call grading its own homework. Four independent, deterministic checks
(`src/pidetect/agent/grounding.py`):

1. **Citation check** — every cited node id must have appeared in a tool result this turn.
2. **Query-argument check** — `tag_function`/`tag_contains` values passed to `find_nodes`/
   `count_nodes` must trace to a prior `list_systems_or_lines()` result. Closes "grounded facts,
   hallucinated query": a wrong tag guess that happens to return an empty (and therefore
   "technically true") result would otherwise pass a check that only inspects output tokens.
3. **Filter-completeness check** — a question naming both a `cls_name` and a `tag_function` must
   be answered from a single call that combined both, not a narrower query that silently dropped
   one constraint.
4. **Judgment-scope check** — a question asking for a subjective/evaluative conclusion ("is this
   well-instrumented?", "which is the most critical instrument?") can never be answered with a
   verdict word, even when every surrounding number is real. A deterministic classifier
   intercepts these before the LLM ever sees the question — the primary defense; this check is
   the second, independent layer in case a future prompt/classifier change ever lets one through.

A failed check doesn't get logged and shipped anyway: the agent loop rejects the answer and either
retries with a corrective message (bounded to 2 attempts) or returns an explicit non-answer —
never the model's last unfinished guess presented as if it were a verified fact.

### Cross-sheet stress-test results

Adversarially built, not a happy-path eval: 73 questions across three real OPEN100 sheets chosen
for different structural profiles — sheet 0 (medium density, mostly-inferred edges), sheet 10
(dense, 236 nodes), sheet 8 (sparse, 102 nodes, mostly-traced edges) — run live against two
different LLM providers (Gemini `gemini-3.1-flash-lite`, Groq `openai/gpt-oss-20b`).
Ground truth computed programmatically from each sheet's real exported graph, never by an LLM
(`docs/phase6_tier1_design/sheet{0,10,8}_stress_questions.json`, `stress_results.md`).

| Sheet | Questions | Aggregate | `not_found` gate | `refuse` gate |
|---|---|---|---|---|
| 0 (medium) | 28 | 25/28 (89%, 1 infra timeout) | 6/6 (**100%**) | 9/9 (**100%**) |
| 10 (dense) | 25 | 24/25 (96%) | 7/8 (87.5%) | 8/8 (**100%**) |
| 8 (sparse) | 20 | 19/20 (95%) | 3/4 (75%) | 8/8 (**100%**) |

`refuse` — declining general-knowledge questions, subjective-judgment questions, and multi-hop
traversal bait — is **100% on all three sheets**, because every one of those triggers a
deterministic, zero-LLM-call classifier that runs before `provider.step()` is ever called, so it
holds identically across providers by construction. `not_found` — correctly reporting absence for
a query that legitimately matches nothing — is **not yet 100% on two of the three sheets**, tracked
below rather than rounded up.

### Known limitations (agent)

- **Compound `not_found` questions can still hit the round cap without answering** (sheet 10 &
  sheet 8's `C3`: "find valves of class X with tag Y" where the combination is empty). The model
  sometimes explores relationally (`get_neighbors` on individual matches) instead of running the
  direct combined-filter query, and never attempts a final answer at all — so the
  retry-on-rejection mechanism never gets a chance to help, since it only intercepts an
  *attempted* answer, not silence. Root-caused, not hidden: `docs/phase6_tier1_design.md`'s
  "Cross-sheet stress hardening" section.
- **The termination guard can over-block a legitimately necessary follow-up call.** A guard was
  added specifically because a model was burning its entire tool budget re-verifying ids it had
  already retrieved (`get_node` calls on results already in hand) instead of concluding. It
  currently can't distinguish "redundant re-verification of a fact already known" from "a
  different kind of information the same question also needs" — confirmed live on a sheet-0
  question needing both a class filter *and* a separate connectivity check. Not fixed yet.
- **Validated within OPEN100 only.** Three sheets, one dataset, one drawing convention, one symbol
  vocabulary, and the fixes above were built to close exactly the failure modes those three sheets
  exposed. No other labeled connectivity-graph dataset exists in this project yet (PID2Graph's own
  eval is blocked on Phase 4's connectivity F1 gate not being met), so generalization to a
  differently-styled P&ID corpus is genuinely untested, not just unclaimed.
- **Tier 2** (multi-hop path-finding, "what's downstream of X", flow simulation) is explicitly out
  of scope for this tier — the agent declines these deterministically rather than chaining
  single-hop lookups and presenting the result as one verified route.

### Running the agent

```bash
# export GEMINI_API_KEY or GROQ_API_KEY first (matches configs/phase6.yaml's `provider` key),
# or set provider: ollama there to use a local model instead.
PYTHONPATH=src python scripts/ask_phase6.py --sheet sheet0 "How many valves are on this sheet?"
PYTHONPATH=src python scripts/ask_phase6.py --sheet sheet10          # interactive REPL, --sheet sheet8 also available
```
Or use the chat panel in the running app (`ChatPanel.jsx`) — the pre-registered `sheet0-demo` /
`sheet10-demo` / `sheet8-demo` jobs (`src/pidetect/api/main.py`) serve the exact locked graphs the
numbers above were measured against, no upload or detection weights needed.

## Architecture

Detection (YOLOv11 + SAHI tiling) → cross-subtype dedupe → symbol erasure → skeleton tracing →
endpoint binding + short-gap bridging → junction (connector/crossing) detection → graph
construction/contraction → instrument-tag OCR (PaddleOCR) → JSON/GraphML export. FastAPI wraps
this as an async job (`POST /jobs` → poll `GET /jobs/{id}` → `GET /jobs/{id}/result`); a minimal
React frontend polls, renders the overlay, and offers JSON/PNG downloads. Full design rationale,
scope cuts, and the latency/hosting analysis behind these choices: `docs/phase5_design.md`.

## Running the demo locally

**Python 3.12 is required for the whole project, not just OCR.** Detection (torch/ultralytics/
sahi) and OCR (paddlepaddle/paddleocr) must run in the *same* interpreter for the API to work as
one process — confirmed this session that all of them install and import together cleanly under
3.12 (paddlepaddle has no wheel for Python 3.13+, which is the only version constraint here).

```bash
python3.12 -m venv .venv312
# Windows: .venv312\Scripts\activate | macOS/Linux: source .venv312/bin/activate
pip install -r requirements.txt
```

You'll also need trained detection weights (`*.pt`, gitignored — train your own via the
Colab/Kaggle workflow below, or point `configs/phase5.yaml`'s `weights_path` at one you have).

**Backend:**
```bash
PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False PYTHONPATH=src \
  uvicorn pidetect.api.main:app --host 127.0.0.1 --port 8000
```
The MKLDNN env var works around a CPU inference crash confirmed on this dev machine
(`NotImplementedError: ConvertPirAttribute2RuntimeAttribute...`, paddlepaddle 3.3.1 CPU/PIR
executor) — see `docs/phase5_design.md` for detail. GPU deployments may not need it.

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Open the printed local URL (typically `http://localhost:5173`), upload a P&ID sheet (e.g. one of
`data/realworld_eval/open100/_raw/*.png` once you've fetched OPEN100), and wait — it's slow
(see Known limitations) but real.

## Data
Public only. The Digitize-PID symbols set (`hamzas/digitize-pid-yolo`) is primary; PID2Graph
OPEN100 (Energy Impact Center) is used for connectivity/OCR evaluation and is what this README's
screenshots are drawn from. **NDA diagrams from the original hackathon project are never
committed, trained on, or used in any screenshot or doc** — confirmed for this repo: nothing
under `data/`, no `*.pt`/`*.onnx`/`*.pth` weights, and no `nda`-named files are tracked in git
(`.gitignore` excludes `/data/`, `/runs/`, weight extensions, and `nda/`/`*_nda*` explicitly).

## Quickstart — rebuild dataset from scratch

```bash
python scripts/build_dataset.py
```

Runs the full Phase 0 pipeline (download → tile → synthetic → merge) with fixed seed 42.
Produces `data/merged/` ready for Phase 1 YOLOv11 training.
Skips steps whose outputs already exist; use `--force` to rebuild everything.

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

## Phase 0 Progress

- [x] 0.1 — Environment & repo baseline (venv, imports verified, git + .gitignore, this checklist)
- [x] 0.2 — Download HF dataset (`hamzas/digitize-pid-yolo`) into `data/`
- [x] 0.3 — Dataset inspection: per-class counts, sample box overlays, tile-size driver
- [x] 0.4 — Tiling pipeline: 640px / 20% overlap → 35 826 tiles across train+val
- [x] 0.5 — Synthetic generator: 200 sheets, 32-class glyph library, YOLO labels + connectivity JSON
- [x] 0.6 — One-command build: `python scripts/build_dataset.py` → train/val/test split, histogram
