# Phase 1.8 Final — Resolution Fix Decision & Phase 2 Go/No-Go

**Eval date:** 2026-06-21 (1.8a/b), 2026-06-24 (1.8c post-retrain)  
**Gate target:** Arrow CtrMt@50% > ~80%, valve + instrument recall intact

---

## A. Branch decision & outcome

| Condition | Value | Threshold | Status |
|:----------|------:|----------:|:-------|
| Arrow CtrMt@50% after 1.8c retrain | 65.2% | ≥ 80% | **Below gate — plateau confirmed** |
| Valve recall | 0.922 | ≥ baseline (0.821) | No regression |
| Instrument recall | 0.994 | ≥ baseline (0.992) | No regression |

**Branch taken:** Retile train set to 320px + retrain (50 epochs, batch=32).

**Outcome:** No improvement. CtrMt@50% for arrows is identical to 1.8b (65.2%).
The 320px retrain hypothesis was that the ceiling was a training/inference resolution
mismatch. That hypothesis is **falsified**. The model trained on 320px tiles sees arrows
at ~32px during training — identical to 320px SAHI inference — but the OPEN100 number
doesn't move. The bottleneck is not model resolution.

---

## B. Phase 1.8 summary — all configurations

| Config | Arrow recall | Arrow CtrMt@50% | Arrow AP@0.5 | Valve recall | Instr recall | Cost |
|:-------|------------:|----------------:|------------:|-------------:|-------------:|-----:|
| 640 whole-tile (baseline) | 0.456 | 57.9% | — | 0.821 | 0.992 | 1× |
| 1280 whole-tile (1.8a) | 0.525 | 64.9% | — | 0.907 | 0.990 | ~2× |
| SAHI slice=320 (1.8b) | 0.493 | 65.2% | 0.337 | 0.922 | 0.994 | 37.5× |
| **320px retrain + SAHI slice=320 (1.8c)** | **0.493** | **65.2%** | **0.337** | **0.922** | **0.994** | **~2×** |

**1.8c result: no improvement over 1.8b.** The plateau at ~65% is not a training/inference
distribution mismatch — the 320px retrain (50 epochs, val mAP50 0.9945) produces identical
OPEN100 Tier-2 metrics to the 640px-trained model with SAHI inference. The ceiling is
elsewhere (see Section D).

---

## C. Kaggle training commands (1.8c)

### Step 1 — Build 320px training dataset

```bash
# On Kaggle, after cloning repo and downloading HF data:
python scripts/build_dataset.py --tile 320 --synth-n 200
# Writes: data/tiled_320/, data/merged_320/, configs/yolo_320.yaml
# Tile count: ~4× current (31K → ~125K tiles; retained by neg_fraction=0.0)
```

### Step 2 — Retrain from small_objects checkpoint

```bash
python -m pidetect.detect.train \
  --data    configs/yolo_320.yaml \
  --model   runs/detect/train_small_objects/weights/best.pt \
  --aug     small_objects \
  --imgsz   640 \
  --epochs  100 \
  --batch   16 \
  --name    train_320tiles
# Saves to: runs/detect/train_320tiles/weights/best.pt
```

`--imgsz 640` is intentional — the model is trained on 320px tiles and Ultralytics
upscales each tile to 640px internally, giving the 2× effective upscale during training
that matches 1.8b inference.

### Step 3 — Evaluate (locally, after pulling weights)

```bash
PYTHONPATH=src .venv/Scripts/python -m pidetect.detect.evaluate \
  --weights runs/detect/train_320tiles/weights/best.pt \
  --realworld --tier open100 \
  --slice-size 320 \
  --extended
# This is the same eval path as 1.8b, but with a model trained at 320px scale.
```

### Production inference (post-1.8c)

```bash
python -m pidetect.detect.predict \
  --weights runs/detect/train_320tiles/weights/best.pt \
  --image   <sheet.png> \
  --slice   320 \
  --imgsz   640
# --slice and --imgsz are now decoupled in predict.py (1.8c infra change).
```

---

## D. Phase 2 go/no-go & arrow diagnosis

**Current status: NOT GO on arrow gate. Valve and instrument: GO.**

| Supercategory | CtrMt@50% | Gate | Status |
|:--------------|----------:|-----:|:-------|
| valve | 97.6% | ≥ 80% | **GO** |
| instrument | 99.4% | ≥ 80% | **GO** |
| arrow | 65.2% | ≥ 80% | **NOT GO** |

**Why the 320px retrain didn't help — candidate root causes:**

1. **OPEN100 ground truth coverage.** The OPEN100 Tier-2 arrow labels were
   annotated at coarse supercategory granularity for real sheets. Arrows with unusual
   styles (filled triangle vs open, very thin signal-line arrows, diagonal arrows on
   flow lines) may be in the GT but absent from the synthetic training distribution
   entirely — a domain gap that no resolution fix can close.

2. **Class imbalance in training.** Arrow instances in the 32-class training set
   (hamzas/digitize-pid-yolo) may be concentrated in one or two sub-classes while
   the OPEN100 sheets contain a wider variety. Per-class AP was not tracked on the
   arrow sub-classes during training.

3. **Annotation quality ceiling.** CtrMt@50% requires a predicted center within
   50% of the GT box size. If 35% of OPEN100 GT arrows are occluded, truncated,
   or labeled at slightly wrong scale, the ceiling is below 80% regardless of model
   quality.

**Recommended next step before escalating to model changes:**
Inspect the 35% miss cases directly — load OPEN100 Tier-2 tiles, overlay GT vs
predictions, classify misses as: (a) model missed a real arrow, (b) GT box is
low-quality, (c) arrow style not in training distribution. This takes one eval
notebook session and determines whether the fix is data, annotation, or architecture.

**Phase 2 scope (pending go):** Fine-grained classifier for:
- idx 16/17/18 — spectacle-blind-like fittings (excluded from valve supercategory)
- idx 3/10 — valve look-alikes in the bowtie/pinch family
- Real-world signal must come from manual annotation or domain augmentation since
  OPEN100 doesn't cover fine-grained valve sub-types

---

## E. Infrastructure changes made in 1.8c prep

| File | Change |
|:-----|:-------|
| [scripts/build_dataset.py](../scripts/build_dataset.py) | Added `--tile N` arg; paths (TILED, SYNTH_TILED, MERGED, YAML) parameterized; `tile=640` backward-compatible |
| [src/pidetect/detect/predict.py](../src/pidetect/detect/predict.py) | Added `--slice` arg; decoupled slice crop size from `--imgsz` in SAHI call |
| [src/pidetect/detect/evaluate.py](../src/pidetect/detect/evaluate.py) | (1.8a/1.8b) `--extended` + `--slice-size` flags already in place |
