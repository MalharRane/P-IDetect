#!/usr/bin/env bash
# PIDetect — Colab/Kaggle GPU training launcher
#
# ── COLAB QUICK START ───────────────────────────────────────────────────────
# Cell 1  (mount Drive once per session for dataset caching):
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Cell 2  (run everything):
#   %cd /content
#   !git clone https://github.com/MalharRane/P-IDetect.git
#   %cd pidetect
#   !bash scripts/colab_setup.sh
#
# ── KAGGLE QUICK START ───────────────────────────────────────────────────────
# Settings → Internet ON.  Then in a notebook cell:
#   %%bash
#   git clone https://github.com/MalharRane/P-IDetect.git
#   cd pidetect
#   bash scripts/colab_setup.sh --no-drive
#
# The dataset rebuilds from scratch each Kaggle session (~15 min).
# To avoid this, save /kaggle/working/pidetect/data to a Kaggle dataset
# and add it as input, then symlink: ln -s /kaggle/input/pidetect-data data
#
# ── DRIVE CACHE STRATEGY (Colab only) ───────────────────────────────────────
# First session  : dataset is built fresh (~15 min) then copied to Drive once.
# Later sessions : rsync from Drive restores it in ~30 s instead of rebuilding.
# Cache lives at: $DRIVE_CACHE  (default: ~/MyDrive/pidetect_cache)
# To force a rebuild:  bash scripts/colab_setup.sh --force-data
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE_CACHE="${DRIVE_CACHE:-/content/drive/MyDrive/pidetect_cache}"
USE_DRIVE=true
FORCE_DATA=false

# Batch size by GPU — edit if you know your runtime in advance.
# T4 16 GB  →  batch=16 (safe), 32 (often fits)
# P100 16 GB → batch=16-32
# V100 16 GB → batch=32
# A100 40 GB → batch=64
BATCH=16

for arg in "$@"; do
  case $arg in
    --no-drive)   USE_DRIVE=false ;;
    --force-data) FORCE_DATA=true ;;
    --batch=*)    BATCH="${arg#*=}" ;;
  esac
done

cd "$REPO_ROOT"
echo "=== PIDetect training launcher ==="
echo "    repo   : $REPO_ROOT"
echo "    batch  : $BATCH"
echo "    drive  : $USE_DRIVE  (cache: $DRIVE_CACHE)"
echo ""

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo "[1/4] Installing requirements..."
pip install -q -r requirements.txt

# ── 2. Dataset: restore from Drive or build ───────────────────────────────────
echo "[2/4] Dataset..."
DRIVE_DATA="$DRIVE_CACHE/data"

if $USE_DRIVE && [ -d "$DRIVE_DATA/merged/images/train" ] && ! $FORCE_DATA; then
  echo "  Restoring from Drive cache (this takes ~30 s) ..."
  mkdir -p data
  rsync -a --info=progress2 "$DRIVE_DATA/" data/
  echo "  Restored."
else
  echo "  Building dataset from scratch (~15 min on first run) ..."
  PYTHONPATH=src python scripts/build_dataset.py
  echo "  Build complete."

  if $USE_DRIVE && [ -d /content/drive ]; then
    echo "  Caching to Drive for future sessions ..."
    mkdir -p "$DRIVE_CACHE"
    rsync -a --info=progress2 data/ "$DRIVE_DATA/"
    echo "  Cached -> $DRIVE_DATA"
  fi
fi

# Sanity-check the dataset is present before spending GPU time
TRAIN_COUNT=$(find data/merged/images/train -name "*.jpg" 2>/dev/null | wc -l)
VAL_COUNT=$(find data/merged/images/val   -name "*.jpg" 2>/dev/null | wc -l)
TEST_COUNT=$(find data/merged/images/test  -name "*.jpg" 2>/dev/null | wc -l)
echo "  tiles: train=$TRAIN_COUNT  val=$VAL_COUNT  test=$TEST_COUNT"
if [ "$TRAIN_COUNT" -lt 100 ]; then
  echo "ERROR: training set looks too small ($TRAIN_COUNT tiles). Aborting." >&2
  exit 1
fi

# ── 3. Train ──────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Training YOLOv11s (epochs=100, imgsz=640, batch=$BATCH) ..."
PYTHONPATH=src python -m pidetect.detect.train \
  --model  yolo11s.pt \
  --data   configs/yolo_baseline.yaml \
  --imgsz  640 \
  --epochs 100 \
  --batch  "$BATCH" \
  --device 0 \
  --degrees 15 \
  --scale   0.5 \
  --mosaic  1.0 \
  --fliplr  0.5

# ── 4. Evaluate on test split ─────────────────────────────────────────────────
echo ""
echo "[4/4] Evaluating on test split ..."
WEIGHTS=$(ls -t runs/detect/train*/weights/best.pt 2>/dev/null | head -1)
if [ -z "$WEIGHTS" ]; then
  echo "ERROR: could not find best.pt under runs/detect/. Training may have failed." >&2
  exit 1
fi

PYTHONPATH=src python -m pidetect.detect.evaluate \
  --weights "$WEIGHTS" \
  --data    configs/yolo_baseline.yaml \
  --split   test \
  --device  0

echo ""
echo "=== Done ==="
echo "  Best weights : $WEIGHTS"
echo "  Eval outputs : $(dirname "$WEIGHTS")/../eval/"
echo ""
echo "  To download weights to your laptop:"
echo "    from google.colab import files"
echo "    files.download('$WEIGHTS')"
echo ""
echo "  Copy the per-class AP table above and send it back with best.pt."
