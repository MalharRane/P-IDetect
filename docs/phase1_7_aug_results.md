# Phase 1.7c — Before/after: scale-focused augmentation

**Baseline:** `runs\detect\train\weights\best.pt` (aug profile: `default`)  
**New:**      `runs\detect\train_small_objects\weights\best.pt` (aug profile: `small_objects`)  
**Eval date:** 2026-06-17  

Diagnosis driving this change: arrow misses are a **pure scale gap** — real OPEN100
flow arrows are ~16 px (diagonal) vs ~79 px in synthetic training data (~5×). The
`small_objects` profile uses `scale=0.9` to cover that range during training.

---

## A. Tier-2 OPEN100 — 3-supercategory AP@50

*(real sheets, out-of-distribution; the honest generalization test)*

| Supercategory | default AP@50 | small_objects AP@50 | Delta |
|:--------------|-------------------:|----------------:|------:|
| valve         |              0.578 |           0.790 | +0.213 |
| arrow         |              0.178 |           0.331 | +0.152 |  ← KEY
| instrument    |              0.979 |           0.989 | +0.010 |

### Arrow detail (Tier-2)

| Model          | n_gt | TP  | Recall | AP@50 |
|:---------------|-----:|----:|-------:|------:|
| default        |  833 | 217 |  0.261 | 0.178 |
| small_objects  |  833 | 380 |  0.456 | 0.331 |

---

## B. In-distribution test split (32-class, data/merged)

*(synthetic test tiles; regression check — should stay ≥ 0.98)*

| Metric              | default | small_objects | Delta |
|:--------------------|----------:|------:|------:|
| mAP@50 (32-class)   |     0.994 | 0.994 | +0.000 |
| mAP@50-95           |     0.985 | 0.982 | -0.003 |
| flow_arrow AP@50    |     0.995 | 0.995 | +0.000 |
| flow_arrow recall   |     0.990 | 0.990 | +0.000 |

---

## Interpretation

**Arrow (Tier-2):** Arrow AP improved by +0.152 — scale jitter is working.  
**Valve (Tier-2):** delta=+0.213 (valve fix requires more than aug alone — see valve root-cause).  
**Regression check:** In-dist mAP held or improved (+0.000) — no regression.
