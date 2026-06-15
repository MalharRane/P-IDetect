# Phase 1 Detection — Error Analysis

**Model:** YOLOv11s, 50 epochs, 640 px tiles, batch 32 (2×T4 DDP)  
**Test split:** 1 428 tiles, real P&IDs only (no synthetic), held out from training  
**Evaluated:** 2026-06-16

---

## Overall test-split metrics

| Metric | Value |
|---|---|
| mAP@50 | **0.994** |
| mAP@50-95 | **0.985** |
| Precision | 0.995 |
| Recall | 0.988 |

Phase 1 gate (**recall ≥ 0.85**): **PASSED** overall. Two individual classes fall below — see below.

---

## 5 weakest classes (test split, sorted by AP@50 then AP@50-95)

| Rank | Class | AP@50 | AP@50-95 | Recall | Notes |
|---|---|---|---|---|---|
| 1 | `gate_valve_handwheel` | 0.980 | 0.960 | **0.865** | Only class below Phase 1 recall gate |
| 2 | `butterfly_valve_partial` | 0.985 | 0.982 | 0.969 | Visually close to other butterfly variants |
| 3 | `gate_valve_actuated_dot` | 0.991 | 0.983 | 0.933 | Gate-valve look-alike family |
| 4 | `gate_valve_actuated_stem` | 0.994 | 0.983 | 0.969 | Gate-valve look-alike family |
| 5 | `flow_arrow` | 0.995 | **0.957** | 0.990 | Weakest localization (mAP50-95); directional arrows vary in shape |

Honourable mentions by mAP@50-95 (localization quality):

| Class | AP@50-95 |
|---|---|
| `tag_rectangle_simple` | 0.955 |
| `flow_arrow` | 0.957 |
| `gate_valve_handwheel` | 0.960 |
| `spectacle_blind_spacer` | 0.967 |
| `tag_rectangle_multiline` | 0.968 |

---

## Most-confused class pairs

From the confusion matrix (see `runs/detect/train/eval_test/confusion_matrix.png`):

1. **gate_valve_handwheel ↔ gate_valve_actuated_stem / gate_valve_actuated_dot**  
   The three actuated gate-valve variants share the same body; the distinguishing feature
   (handwheel vs stem cap vs dot cap) is small and orientation-dependent. Recall on
   `gate_valve_handwheel` (0.865) is far below the family average (~0.97), suggesting
   the model sometimes detects the symbol but misclassifies it as one of the sibling classes.

2. **butterfly_valve_partial ↔ butterfly_valve_open / butterfly_valve_open_v2**  
   Partial-open butterfly valves look like a rotated version of the fully-open symbol.
   Recall 0.969 and AP@50 0.985 both sit below the butterfly-valve family average (~0.995),
   consistent with inter-family confusion at tile boundaries.

3. **flow_arrow — missed detections (vs background)**  
   Recall is 0.990 but AP@50-95 is only 0.957 — the weakest localization score across all
   classes. Flow arrows are thin directional lines whose bounding boxes are ambiguous (the
   "right" box extent is ill-defined), causing loose regression and a steep IoU drop-off.

The overall confusion matrix diagonal is near-perfect; the only notable off-diagonal mass is
within the gate-valve sub-family.

---

## OBB (YOLOv11-OBB) assessment

**Not worth it yet.** Reasons:

- Our public training data (HF `hamzas/digitize-pid-yolo` + synthetic) is **axis-aligned only**;
  no oriented-box angle labels exist.
- Adding OBB requires either: (a) re-labelling all source images with angle annotations, or
  (b) generating synthetic sheets with rotation and auto-producing oriented labels — which our
  synth generator can do (it already places glyphs at random rotation) but the label format
  would need to change from YOLO bbox to YOLO OBB.
- The current mAP@50 of **0.994** on the synthetic-distribution test set is already strong;
  the mAP@50 vs mAP@50-95 gap (0.009) is small, indicating boxes are well-localised for
  upright symbols.
- Revisit OBB after the real-world eval set is built — if real sheets with rotated symbols
  show a meaningful recall drop, that is the trigger to invest in OBB labels + retraining.

---

## Recommended next step

**Build the real-world eval set** (`data/realworld_eval/`) before changing the model.

Rationale: the current test split is drawn from the same HF dataset as training — same scanner
vendor, same symbol library, same scale. A 0.994 mAP on this split could be inflated by
distribution similarity. We do not yet know how the model behaves on a genuinely out-of-
distribution P&ID (different company, hand-drawn symbols, faded scans).

Concretely:
1. Hand-label 5–10 real P&ID sheets from a public source (see `data/realworld_eval/README.md`).
2. Run `evaluate.py --realworld` to get an honest out-of-distribution AP table.
3. **If real-world recall drops below 0.7:** scale up to yolo11m + more synth instances for
   weak classes.  
   **If real-world recall stays above 0.85:** proceed directly to Phase 2 (fine-grained
   classifier for the gate-valve / butterfly-valve look-alike families).

The single concrete model improvement that is already clearly warranted regardless of
real-world results: **generate 3–5× more synthetic instances for `gate_valve_handwheel`**
(the only class below the recall gate at 0.865) before the next training run.
