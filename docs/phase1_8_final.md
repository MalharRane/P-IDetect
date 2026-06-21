# Phase 1.8 Final — Resolution Fix Decision & Phase 2 Go/No-Go

**Eval date:** 2026-06-21  
**Gate target:** Arrow CtrMt@50% > ~80%, valve + instrument recall intact

---

## A. Branch decision

| Condition | Value | Threshold | Status |
|:----------|------:|----------:|:-------|
| Arrow CtrMt@50% (best so far) | 65.2% | ≥ 80% | **Below — retrain needed** |
| Valve recall | 0.922 | ≥ baseline (0.821) | No regression |
| Instrument recall | 0.994 | ≥ baseline (0.992) | No regression |

**Branch: retile train set to 320px + retrain.**  
Two-pass inference is NOT needed — large symbols held at 320px slice in 1.8b.

**Why the 65% ceiling exists:** 1.8a and 1.8b both plateau at ~65% because the model
was trained on 640px tiles where flow arrows appear ~16px. At inference time (imgsz=1280
or SAHI slice=320) it sees arrows at ~32px — out-of-distribution. The fix is to train on
320px tiles at imgsz=640 so training and inference are consistent at 2× upscale.

---

## B. Phase 1.8 summary — all configurations

| Config | Arrow recall | Arrow CtrMt@50% | Arrow prec | Valve recall | Instr recall | Cost |
|:-------|------------:|----------------:|-----------:|-------------:|-------------:|-----:|
| 640 whole-tile (baseline) | 0.456 | 57.9% | 0.696 | 0.821 | 0.992 | 1× |
| 1280 whole-tile (1.8a) | 0.525 | 64.9% | 0.722 | 0.907 | 0.990 | ~2× |
| SAHI slice=320 (1.8b) | 0.493 | 65.2% | 0.667 | 0.922 | 0.994 | 37.5× |
| **320px retrain (1.8c)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **~2×** |

1.8c target: inference at `--slice 320 --imgsz 640`, same 1.8b eval path.

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

## D. Phase 2 go/no-go

**Current status: NOT GO.**

Arrow CtrMt@50% = 65.2% (best: 1.8b sahi-320) is below the ~80% gate. Valve and
instrument recall are intact and actually improved. The blocker is the training/inference
distribution mismatch for arrows — addressed by 1.8c retrain on 320px tiles.

**Becomes GO when:** After 1.8c Kaggle training, re-eval at `--slice-size 320` shows
arrow CtrMt@50% ≥ 80% with valve recall ≥ 0.821 and instrument recall ≥ 0.992.

**Phase 2 scope (pending go):** Fine-grained classifier for:
- idx 16/17/18 — spectacle-blind-like fittings (excluded from valve supercategory; need own class)
- idx 3/10 — valve look-alikes in the bowtie/pinch family
- Initial step: build a dedicated classifier for these families; OPEN100 doesn't cover them so
  real-world signal must come from manual annotation or domain-specific augmentation

---

## E. Infrastructure changes made in 1.8c prep

| File | Change |
|:-----|:-------|
| [scripts/build_dataset.py](../scripts/build_dataset.py) | Added `--tile N` arg; paths (TILED, SYNTH_TILED, MERGED, YAML) parameterized; `tile=640` backward-compatible |
| [src/pidetect/detect/predict.py](../src/pidetect/detect/predict.py) | Added `--slice` arg; decoupled slice crop size from `--imgsz` in SAHI call |
| [src/pidetect/detect/evaluate.py](../src/pidetect/detect/evaluate.py) | (1.8a/1.8b) `--extended` + `--slice-size` flags already in place |
