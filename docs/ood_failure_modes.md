# OOD Failure-Mode Diagnosis (subtask 1.7a)

**Model:** `runs/detect/train/weights/best.pt`  
**Eval set:** OPEN100 Tier 2 (12 real sheets, 197 tiles)  
**Date:** 2026-06-17  
**Inference conf threshold:** 0.01 (low, to avoid masking B-bucket cases)  

Bucket definitions:
- **TP** — correct supercategory prediction, IoU ≥ 0.5
- **A — NO_FIRE** — no prediction overlaps GT above IoU 0.1 (true miss)
- **B — MISLOCATED** — prediction exists nearby but IoU < 0.5
- **C — WRONG_CLASS** — well-localized (IoU ≥ 0.5) but different supercategory
- **D — EXCLUDED_IDX** — well-localized, predicted index is one we excluded from
  the supercategory mapping (14/15/16/17/18/24 for valve). **Measurement artifact.**

---

## Valve failure breakdown

| Bucket                               | Count |      % |
|--------------------------------------|------:|-------:|
| TP (matched)                         |   348 |    75.2% |
| A — NO_FIRE                          |    26 |     5.6% |
| B — MISLOCATED                       |    62 |    13.4% |
| C — WRONG_CLASS                      |    19 |     4.1% |
| D — EXCLUDED_IDX (artifact)          |     8 |     1.7% |
| **Total**                            |   463 |     100% |

**Artifact vs real (non-TP only):**
  D (artifact):      8 / 115  (7.0%)
  A+B+C (real):    107 / 115  (93.0%)

Wrong-class predicted indices (bucket C, valve GT):
  idx 21  spring_element                × 10
  idx 31  tag_rectangle_multiline       × 4
  idx 22  heat_exchanger_strainer       × 2
  idx 26  instrument_bubble_medium      × 2
  idx 25  instrument_bubble_large       × 1

---

## Arrow failure breakdown

| Bucket                               | Count |      % |
|--------------------------------------|------:|-------:|
| TP (matched)                         |   316 |    37.9% |
| A — NO_FIRE                          |   291 |    34.9% |
| B — MISLOCATED                       |   219 |    26.3% |
| C — WRONG_CLASS                      |     7 |     0.8% |
| D — EXCLUDED_IDX (artifact)          |     0 |     0.0% |
| **Total**                            |   833 |     100% |

**Artifact vs real (non-TP only):**
  D (artifact):      0 / 517  (0.0%)
  A+B+C (real):    517 / 517  (100.0%)

Wrong-class predicted indices (bucket C, arrow GT):
  idx 15  ball_valve                    × 3
  idx 19  reducer                       × 2
  idx 20  spectacle_blind_spacer        × 2

---

## Key question

**How much of the valve AP drop is bucket D (artifact) vs A/B/C (real)?**

  D (artifact):      8 / 115  (7.0%)
  A+B+C (real):    107 / 115  (93.0%)

Interpretation: bucket D cases are where the model found something at the right
location but our supercategory mapping deliberately excluded the predicted index
(spectacle-blind family 16/17/18 and ambiguous circles 14/15/24). These count as
misses in AP scoring even though the model 'saw' something there. A/B/C cases are
genuine generalization failures — the model either missed the symbol entirely (A),
roughly localized it but not well enough (B), or confused it with a different class (C).

---

## Annotated examples

See `docs/ood_examples/` (8 tiles).  
Green box = GT. Red/orange box = best prediction. Bucket label in lower-left corner.
