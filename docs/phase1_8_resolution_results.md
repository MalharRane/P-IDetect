# Phase 1.8b — Higher-Res SAHI Slicing (slice=320, imgsz=640)

**Weights:** `runs/detect/train_small_objects/weights/best.pt` (unchanged)  
**Eval date:** 2026-06-21  
**Tier:** Tier-2 OPEN100 tiles (197 pre-sliced 640×640 tiles, 12 sheets)

**Approach:** Each existing 640×640 eval tile is re-sliced by SAHI to 320×320 sub-tiles
(overlap=0.2, step=256px → 3 positions per axis → 9 sub-tiles per tile). Ultralytics
resizes each 320px sub-tile to imgsz=640, giving the same ~2× effective upscale as 1.8a.
GREEDYNMM merges cross-sub-tile detections back to the 640px tile's coordinate space.
No weights changed, no data rebuilt.

---

## A. Supercategory AP — Three-way comparison

| Supercategory | AP@0.5 (640) | AP@0.5 (1280) | AP@0.5 (sahi-320) | AP@0.3 (640) | AP@0.3 (1280) | AP@0.3 (sahi-320) |
|:--------------|-------------:|--------------:|------------------:|-------------:|--------------:|------------------:|
| valve         |        0.790 |         0.877 |             0.900 |        0.935 |         0.962 |             0.965 |
| **arrow**     |    **0.331** |     **0.391** |         **0.337** |    **0.519** |     **0.584** |         **0.558** |
| instrument    |        0.989 |         0.987 |             0.989 |        0.994 |         0.987 |             0.991 |

---

## B. Extended task-relevant metrics

| Config | n_gt | n_tp | Recall | Prec | CtrMt@25% | CtrMt@50% |
|:---|---:|---:|---:|---:|---:|---:|
| **Arrow** | | | | | | |
| 640 (baseline) | 833 | 380 | 0.456 | 0.696 | 55.0% | 57.9% |
| 1280 (1.8a)    | 833 | 437 | 0.525 | 0.722 | 63.6% | 64.9% |
| sahi-320 (1.8b)| 833 | 411 | 0.493 | 0.667 | 61.3% | **65.2%** |
| **Valve** | | | | | | |
| 640 (baseline) | 463 | 380 | 0.821 | 0.727 | 80.3% | 95.0% |
| 1280 (1.8a)    | 463 | 420 | 0.907 | 0.771 | 90.3% | 97.6% |
| sahi-320 (1.8b)| 463 | 427 | 0.922 | 0.740 | 90.3% | 97.6% |
| **Instrument** | | | | | | |
| 640 (baseline) | 619 | 614 | 0.992 | 0.908 | 99.5% | 99.7% |
| 1280 (1.8a)    | 619 | 613 | 0.990 | 0.898 | 99.0% | 99.0% |
| sahi-320 (1.8b)| 619 | 615 | 0.994 | 0.834 | 99.4% | 99.4% |

---

## C. Fragmentation check (mandatory gate)

| Supercategory | Recall (640) | Recall (sahi-320) | Delta | Regressed? |
|:--------------|-------------:|------------------:|------:|:-----------|
| valve         |        0.821 |             0.922 | +0.101 | **No — improved** |
| instrument    |        0.992 |             0.994 | +0.002 | **No — flat** |

SAHI's GREEDYNMM merges sub-tile detections correctly for valve and instrument — no
fragmentation for symbols that fit within a 320px slice (valves ~30-50px, instruments
~40px). Gate passed.

---

## D. Deployment budget

| Config | Model calls | Wall-clock (197 tiles) | Relative cost |
|:-------|------------:|-----------------------:|--------------:|
| 640 whole-tile | 197 | 26.4 s | 1× |
| 1280 whole-tile (1.8a) | 197 | ~53 s est. | ~2× |
| sahi-320 (1.8b) | **1,773** | **990 s (16.6 min)** | **37.5×** |

SAHI sub-tile count: 9 sub-tiles per 640px tile (3 positions × 3 positions per axis at
step=256px). Wall-clock increase is 37.5× — much larger than the 9× call-count ratio —
because each SAHI call adds Python-level image-crop + NMM overhead per tile.

---

## E. Interpretation

### Arrow CtrMt@50%: HELD through SAHI (+7.3pp vs baseline)

Both methods of delivering the 2× effective upscale (inference resize vs. SAHI
sub-slicing) produce the same CtrMt@50% improvement: 57.9% → 64.9% (1.8a) vs
57.9% → 65.2% (1.8b). This is consistent — it is the upscale itself that helps,
not the mechanism.

### But AP@0.5 and recall diverge from 1.8a

| Metric | 640 | 1280 (1.8a) | sahi-320 (1.8b) |
|:---|---:|---:|---:|
| Arrow recall | 0.456 | 0.525 | 0.493 |
| Arrow precision | 0.696 | 0.722 | 0.667 |
| Arrow AP@0.5 | 0.331 | 0.391 | 0.337 |

Recall (0.493) and AP@0.5 (0.337) are worse than 1.8a, and precision dropped below
baseline. Cause: SAHI NMM operates at fine granularity on 320px crops of a
640px tile. Arrows near sub-tile boundaries may generate duplicate or split boxes that
GREEDYNMM fails to merge cleanly at IoU≥0.5, hurting AP@0.5 even as CtrMt@50%
(which only needs one center to land within the GT box) improves.

CtrMt@50% is the more robust metric here because it is immune to duplicate-box
penalty — it only asks whether the GT symbol's centre was "found", not how cleanly.

### Why 1.8a wins for the arrow gain

1.8a (imgsz=1280) applies the 2× upscale at the Ultralytics level (resize the entire
640px tile to 1280px, single forward pass, no seam effects). This is NMM-free, so it
picks up more TP (437 vs 411) with higher precision (0.722 vs 0.667). 1.8b pays an
unnecessary NMM tax for the same resolution benefit.

### Conclusion and next step

**Gate met:** Arrow CtrMt@50% improved by +7.3pp via SAHI sub-slicing; valve and
instrument held or improved. The resolution fix is confirmed through both delivery
mechanisms.

**However, 1.8b is not the deployment path:** 37.5× wall-clock cost for a result
that 1.8a already achieves more cheaply. The correct deployment fix is to retrain with
larger tiles so training matches inference at the native scale — no inference-time
workaround needed. This is Phase 1.8c: retrain with `imgsz=1280` (or re-tile at 1280px
crop size) and re-evaluate.

Target for 1.8c: arrow CtrMt@50% > 70% with no deployment-cost penalty vs current.
