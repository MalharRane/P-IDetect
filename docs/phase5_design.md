# Phase 5 Design — Deployable Demo (Scope Only)

**Status:** Scope proposal. No code written. Read alongside `docs/phase4_final.md` (connectivity
frozen at mean F1=0.549, gate not met) and `docs/phase3_results.md` (OCR gate passes: 90.7%
ok-rows, 83.8% overall).

Target: upload a P&ID sheet → detected symbols + instrument tags + connectivity graph, viewable
in-browser, downloadable as JSON. The risk this document is written to control is scope creep —
every "nice to have" below is explicitly cut, not silently deferred.

---

## 0. Runtime environment (resolved, not optional)

This session's Phase 3 work discovered a real constraint that must be settled before any FastAPI
code is written: on this dev machine, the default Python is 3.14, which has **no paddlepaddle
wheel**, forcing a separate Python 3.12 venv (`.venv312`) for OCR while detection
(torch/ultralytics/sahi) stayed on the 3.14 venv. That split is a **local dev-machine workaround
only** — it must not leak into the deployed architecture as two processes/venvs glued together.

**Decision:** the deployed backend pins **Python 3.12** everywhere. Confirmed this session:
torch 2.x, ultralytics, and sahi all install cleanly under 3.12 (they have no floor above it),
and it's the version paddlepaddle/paddleocr already require. One interpreter, one venv, one
FastAPI process holding both the YOLO/SAHI model and the PaddleOCR engine in memory. No
subprocess boundary, no IPC, no second service — that would be complexity Phase 5 doesn't need.

---

## 1. Minimum viable demo

**Flow:** upload sheet → background job runs the existing pipeline → poll status → view result
(overlay + tags + graph) in browser → download JSON.

**Backend (FastAPI), 4 endpoints, nothing else:**

| Endpoint | Purpose |
|---|---|
| `POST /sheets` | Accept one image upload, save it, start the pipeline as a background job, return `job_id`. |
| `GET /jobs/{job_id}` | Return status (`queued`/`running`/`done`/`failed`) + coarse stage label (see §2). |
| `GET /jobs/{job_id}/result` | Once done: JSON with detected symbol boxes+classes, instrument tags, and the contracted graph (nodes+edges) — i.e. `run_step8`'s existing JSON export, plus the tag fields already wired into it (`docs/phase3_design.md` §5, verified in this repo's `export.py`). |
| `GET /sheets/{job_id}/image` | Serve the original uploaded sheet (for the frontend to draw the overlay on). |

Job state: an in-process dict (`job_id -> {status, stage, result_path}`), driven by FastAPI
`BackgroundTasks`. **Explicitly not** Celery/Redis/a task queue — a demo serving one user at a
time doesn't need a broker, and adding one only grows the hosting footprint problem (§3) for zero
demo value. Known v1 limitation, not fixed: concurrent uploads serialize behind one worker. Fine
for a demo, wrong for production — out of scope here.

**Pipeline invoked, unchanged:** steps 0–8 exactly as `scripts/run_phase4_steps03.py` already
orchestrates them (SAHI slice=320 detection → ROI filter → cross-class NMS → node set → OCR
(`run_ocr_on_nodes`) → erase → skeleton → junction analysis → short-gap veto → graph construction
→ contraction → arrow direction → export). **Step 9 (eval-vs-GT) is dropped** — there is no GT for
a user-uploaded sheet. Nothing else about the pipeline changes for Phase 5; this phase wraps it,
it doesn't touch it.

**Frontend (React), 3 views, nothing else:**
1. Upload screen (drag-drop or file picker) + a progress indicator while the job runs.
2. Result screen: the sheet image with an SVG/canvas overlay — colored boxes per node type
   (matching the existing diagnostic-render palette in `run_phase4_steps03.py`), tag text next to
   instrument bubbles, edges drawn as lines between node centroids. Pan/zoom (sheets are
   1700–2800px on the OPEN100 eval set, real production sheets larger per CLAUDE.md — pan/zoom is
   not optional).
3. A "Download JSON" button on the result screen. That's the entire export UI — no GraphML
   download, no CSV, no batch upload. If someone wants GraphML they can hit the file on disk;
   don't build a second export path into the UI for v1.

**Explicitly cut from v1** (revisit only if the above ships and there's time left):
- Multi-sheet / batch upload
- Any editing of results (correcting a box, re-typing a misread tag)
- User accounts, history of past uploads, anything persistent beyond one job's lifetime
- A separate node-link graph view (§5)
- GraphML download, CSV export, any format beyond the one JSON already produced
- Confidence-threshold sliders or any other inference knob exposed in the UI
- Mobile-responsive layout — this is a desktop-viewed technical diagram tool

---

## 2. Latency — measured, not estimated

Measured on this machine (CPU-only torch build, `torch.cuda.is_available() == False`) against
real OPEN100 sheets (1744×2604 and 1860×2792px — smaller than the 5000–7000px full sheets
CLAUDE.md describes as the eventual production target; see caveat below).

| Stage | Measured | Notes |
|---|---|---|
| SAHI detection (slice=320, imgsz=640) | **9.8–19.9s** (2 sheets) | Includes per-call model construction; a warm in-process model would sit at the low end of this range per request. |
| Graph pipeline (erase→skeleton→junction→contract→export, cached detections) | **~6.9s** | Steps 0–8 minus detection, sheet 0. |
| OCR engine construction | **3.6s** | One-time, at server startup — not per-request if the engine is held in memory across requests. |
| OCR per instrument bubble | **2.7s/bubble** | Sheet 0, 36 bubbles → **97.7s total for OCR alone.** |

**Total per sheet, warm server: roughly 115–125 seconds**, of which OCR is ~80%. This is the real
number — not a guess — and it rules out a plain synchronous request: most reverse proxies and
several free hosts cap a single HTTP request around 30–60s, and a 2-minute blocking spinner is a
bad demo experience regardless.

**Decision: async job-queue with polling** (§1's 4-endpoint design already assumes this — it's
why `POST /sheets` returns a `job_id` instead of a result). The frontend polls `GET /jobs/{id}`
every ~2s and shows a coarse stage label (`detecting symbols` → `reading tags` → `building
graph`) rather than a fake percentage — real per-bubble OCR progress is knowable (bubble N of M)
and worth surfacing; per-tile detection progress is not worth the wiring for a demo.

**Caveat to state to viewers, not solve now:** OCR's 2.7s/bubble is almost certainly per-call
model-dispatch overhead on tiny crops, not raw compute — PaddleOCR supports batched `.predict()`
calls, which would likely cut this substantially. That's a real, identified optimization, but it's
implementation, not scope — not undertaken in this design pass. **Real production sheets are also
2–4x larger per side than OPEN100** (CLAUDE.md: 5000–7000px vs. the ~1700–2800px measured here),
which for slice=320 tiling means roughly 4–16x more tiles → detection time could realistically run
into minutes, not seconds, on a full-size sheet. State this range to anyone watching the demo;
don't imply the measured number generalizes to full-size production sheets untested.

---

## 3. Hosting reality

**Dependency footprint, measured on this machine:**

| Component | Size |
|---|---|
| torch + ultralytics + sahi (detection) | ~550MB |
| paddlepaddle + paddleocr + paddlex (OCR) | ~790MB |
| PaddleOCR model weights (det+rec+doc-orientation, downloaded on first use) | 171MB |
| YOLO weights | 19MB |
| **Total** | **~1.5GB of installed dependencies**, before counting base OS/Python/runtime memory |

Both deep-learning frameworks are required simultaneously — detection needs torch, OCR needs
paddle, and there's no way to drop either without cutting a core Phase-5 deliverable (symbols or
tags). Loaded in memory at once (YOLO model + PaddleOCR det/rec/doc-orientation models resident
for the process lifetime, per §1's "hold in memory" design), realistic RSS is comfortably above
1GB and likely higher under load.

**Assessment against common free/cheap tiers:** most free web-service tiers (Render free, Railway
free trial, Fly.io free allowance) cap RAM at 256MB–1GB and aggressively sleep/cold-start idle
services — a cold start alone would mean re-downloading or re-loading ~1.5GB of frameworks plus
171MB of OCR models before the first request can even begin, on top of the ~2-minute
per-sheet inference time already measured. A serverless/FaaS-style host is worse: most impose
hard request timeouts (10–60s) that the measured ~2-minute pipeline blows through regardless of
the async design in §2 (the job still has to run somewhere, and that somewhere needs to survive
past the request that started it). Hugging Face Spaces (free CPU tier, ~16GB RAM, long-running
container rather than per-request FaaS) is the one free option that plausibly fits this footprint
and latency — but that's untested here, and "plausibly fits" is not the same as verified working.

**Recommendation: local demo + recorded video + written deployment instructions is the v1
deliverable.** A working local demo, walked through on camera, is honest and complete; a
deployed link that times out, cold-starts for a minute, or falls over under the dependency weight
actively undersells the work. Ship: (1) a one-command local run (`docker-compose up` or an
equivalent documented setup covering both the 3.12 backend venv and the React dev server), (2) a
short screen-recording of the golden path (upload → wait → overlay + tags + graph → JSON
download) on a couple of OPEN100 sheets, (3) written instructions for anyone who wants to run it
themselves. A Hugging Face Spaces deployment is a reasonable **stretch goal after** the local demo
and video exist — attempt it if there's time left, don't block the v1 deliverable on it, and don't
promise it in advance.

---

## 4. React shell reuse

Searched this repo for any carried-over hackathon frontend (`*.tsx`/`*.jsx`/`package.json`/React
or Flask remnants) — **none exists here.** CLAUDE.md's "we may reuse a React UI shell... later"
refers to something in the original (NDA'd) hackathon repo, which is not part of this from-scratch
rebuild and was never committed here (correctly — hackathon diagrams/code are under NDA per
CLAUDE.md's hard rules). If that shell exists somewhere the user can point to, it's worth a quick
look before starting; absent that, **v1 builds fresh, deliberately minimal** — 3 views (§1), no
component library beyond whatever's needed for pan/zoom and basic layout (e.g. a single
lightweight canvas/SVG viewer library, not a full design system). Do not resurrect Streamlit —
CLAUDE.md already decided FastAPI + React over Streamlit for this rebuild.

---

## 5. Graph visualization

**Decision: overlay on the sheet image, not a separate node-link view, for v1.**

Reasoning: the contracted graphs measured this session run ~150–200 nodes and ~150 edges per
sheet (sheet 0: 158 nodes, 147 edges, post-contraction). A force-directed node-link layout at that
size, decontextualized from the original drawing, reads as noise to a non-expert — there's no way
for a viewer to check it against anything they recognize. An overlay drawn directly on the
original P&ID (colored boxes per node type, tag text at instrument bubbles, lines for edges) lets
a viewer visually cross-check the system's output against the diagram they can already see and
understand — that's the more convincing demo for someone who isn't a P&ID expert, and it's also
the most honest presentation given connectivity's frozen gate-not-met state (mean F1=0.549,
`docs/phase4_final.md`): showing edges in place, on the real drawing, next to their real context
makes both correct and incorrect edges visually checkable, rather than hiding graph quality behind
an abstract layout algorithm's aesthetics.

A separate node-link view (e.g. via a lightweight graph-rendering library) is a plausible v2
addition if the overlay ships and there's appetite for it, but it does not gate v1 — cut per the
scope-creep concern this whole document is written against.

---

## Summary — what v1 actually is

Upload → FastAPI background job (Python 3.12, one process, both frameworks resident in memory) →
poll for status with a coarse stage label → sheet image with an SVG/canvas overlay (boxes + tags +
edges) → JSON download. ~2 minutes per sheet on CPU, dominated by per-bubble OCR (measured, not
estimated) — surfaced to the user as a progress indicator, not hidden behind a spinner. Shipped
as a local demo + a recorded walkthrough + written setup instructions; a hosted deployment
(Hugging Face Spaces being the most plausible free fit) is an explicit stretch goal, not the
committed deliverable, given the ~1.5GB dual-framework dependency footprint measured above.
