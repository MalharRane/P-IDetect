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

**Updated 2026-06-24 (post-triage). Arrows: ACCEPT-AND-DOCUMENT. Phase 2 GO.**

| Supercategory | CtrMt@50% | Gate | Status |
|:--------------|----------:|-----:|:-------|
| valve | 97.6% | ≥ 80% | **GO** |
| instrument | 99.4% | ≥ 80% | **GO** |
| arrow | 65.2% | ≥ 80% | **accept-and-document** |

**Arrow triage completed — see `docs/arrow_triage/miss_breakdown.md`.**

**Question 1 — Is OPEN100 'arrow' the same class as our flow_arrow?**  
YES. Visual inspection of 30 GT crops confirms all are solid filled-triangle arrowheads
on solid process lines — identical semantically to idx-23 `flow_arrow`. Bucket (d)
(semantic mismatch) contributes **0%** to the miss rate. The eval is apples-to-apples.

**Question 2 — Why are the ~35% missed?**  
Root cause is a pure **training-distribution scale gap** (bucket c):

| Bucket | Description | Est. % of missed |
|:---|:---|---:|
| (c) Training distribution gap | 99.8% of OPEN100 arrows are 8–30px; synth median is 79.2px | ~100% |
| (b) GT quality issue | 0 arrows below 8px, 0 near edge | 0% |
| (d) Semantic mismatch | confirmed 0% by visual inspection | 0% |
| (a) Genuine model miss | 1 arrow (30px) in detectable range | <1% |

All 451 OPEN100 arrows fall in 8.8–30.0 px diagonal (median 17.8px). The synthetic
training distribution has zero overlap with this range (synth median 79.2px, 4.4× larger).
The 1.8c retrain confirmed this: changing training tile resolution does not change the
size of synthetic arrows relative to the tile, so the scale gap is unaffected.

**Gate decision:**
- Arrow gate is formally not met (65.2% vs 80% target)
- Root cause is training data, not model architecture
- Fix: add real arrow crops as synth templates from a held-out OPEN100 slice — low effort,
  no full retrain needed. Not a P2 prerequisite.
- **Phase 2 starts immediately. Arrows are a separate, lower-criticality track.**

**Phase 2 scope:** Fine-grained classifier for:
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
