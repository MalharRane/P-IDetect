# Phase 1.8a — Arrow Resolution Probe (imgsz=640 vs 1280)

**Weights:** `runs/detect/train_small_objects/weights/best.pt` (unchanged from 1.7d)  
**Eval date:** 2026-06-21  
**Tier:** Tier-2 OPEN100 tiles (same tiles used in 1.7c/d; no retraining, no re-tiling)

**Hypothesis:** Real P&ID flow arrows (~16 px diagonal) are below the model's effective
detection resolution at imgsz=640. A clean 2× upscale to imgsz=1280 should lift arrow
recall without a precision penalty if resolution is the bottleneck.

---

## A. Side-by-side: imgsz=640 vs imgsz=1280

### Supercategory AP

| Supercategory | AP@0.5 (640) | AP@0.5 (1280) | Delta | AP@0.3 (640) | AP@0.3 (1280) | Delta |
|:--------------|-------------:|--------------:|------:|-------------:|--------------:|------:|
| valve         |        0.790 |         0.877 | +0.087 |        0.935 |         0.962 | +0.027 |
| arrow         |        0.331 |         0.391 | +0.060 |        0.519 |         0.584 | +0.065 |
| instrument    |        0.989 |         0.987 | −0.002 |        0.994 |         0.987 | −0.007 |

### Extended task-relevant metrics

| Supercategory | n_gt | n_tp (640) | Recall (640) | Prec (640) | n_tp (1280) | Recall (1280) | Prec (1280) | Δ Recall | Δ Prec |
|:--------------|-----:|-----------:|-------------:|-----------:|------------:|--------------:|------------:|---------:|-------:|
| valve         |  463 |        380 |        0.821 |      0.727 |         420 |         0.907 |       0.771 |   +0.086 | +0.044 |
| **arrow**     |**833**| **380**  |    **0.456** |  **0.696** |     **437** |     **0.525** |   **0.722** | **+0.069** | **+0.026** |
| instrument    |  619 |        614 |        0.992 |      0.908 |         613 |         0.990 |       0.898 |   −0.002 | −0.010 |

### Center-match (Phase 4 connectivity metric)

| Supercategory | CtrMt@25% (640) | CtrMt@50% (640) | CtrMt@25% (1280) | CtrMt@50% (1280) | Δ CtrMt@50% |
|:--------------|----------------:|----------------:|-----------------:|-----------------:|------------:|
| valve         |           80.3% |           95.0% |            90.3% |            97.6% |      +2.6pp |
| **arrow**     |       **55.0%** |       **57.9%** |        **63.6%** |        **64.9%** |   **+7.0pp** |
| instrument    |           99.5% |           99.7% |            99.0% |            99.0% |      −0.7pp |

---

## B. Interpretation

**Resolution hypothesis: CONFIRMED — but only partially.**

Doubling imgsz from 640→1280 delivers:

- Arrow recall: **+0.069** (+15% relative, 380→437 TPs out of 833 GT)
- Arrow CtrMt@50%: **+7.0 pp** (57.9%→64.9%)
- Arrow precision: **+0.026** — no FP explosion; precision *improved*
- Valve: recall +0.086, no regression

Instrument is flat (−0.002 recall) — resolution was never the bottleneck there.

**Why "only partially":** Arrow recall at 1280 is still only 52.5%, leaving ~390 arrows
completely missed. Resolution was a real bottleneck, but not the sole cause of the
1.7d arrow gap. At least two additional factors remain:

1. **Training tile size mismatch** — the `data/merged` tiles were generated at 640 px.
   At inference imgsz=1280, the model sees features at a scale it was never trained on.
   The gain we see is despite this mismatch, not because of it.
2. **Domain gap** — training is entirely synthetic (clean lines, consistent arrow shapes).
   Real P&ID arrows vary in style, stroke width, and rendering quality in ways that
   larger input doesn't fix.

**Precision held** — upscaling did not trigger the false-positive explosion that would
contradict the resolution hypothesis. This is consistent with the model genuinely
resolving more true arrows, not just lowering its threshold.

---

## C. Decision

The +7pp CtrMt@50% payoff at zero training cost confirms the resolution hypothesis is
real and actionable. Escalate to **1.8b**: retrain with `imgsz=1280` (or equivalently
re-tile at native size so arrows render larger during training, matching the inference
upscale). The target is CtrMt@50% > 70% for arrows.

Do not lower conf threshold as a substitute — that trades precision for recall and will
produce noisy graph nodes in Phase 4.
