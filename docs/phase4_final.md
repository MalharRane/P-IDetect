# Phase 4 — Final Frozen Record

**Date frozen:** 2026-07-19 (re-frozen same day after the node-dedupe correction, §2)
**Status:** Connectivity work FROZEN at this state. No further tuning planned until the
upstream levers named in §5 (dominant residual) or the open items in §6 are picked up as new work.

**Config:** `configs/phase4.yaml` as committed alongside this doc —
`mask_all_nodes: true` (per-pair third-party masking), `mask_all_nodes_fallback: false`,
`min_branch_len_px: 15`, `short_gap_stub_angle_tol_deg: 55`.
**Node set:** `src/pidetect/graph/erase.py` — `centroid_nms(..., group_key=scored_family_group_key)`
(valve + instrument_bubble* family-scoped dedup, §2) + `assert_no_duplicate_scored_nodes()`
construction-time regression guard.
**Sheets evaluated:** 0, 3, 10 (OPEN100)

---

## 1. Frozen Metrics

| Sheet | Match% | TP | FP | FN | P | R | F1 | Cross% |
|---|---|---|---|---|---|---|---|---|
| 0 | 98.6% | 23 | 27 | 8 | 0.460 | 0.742 | 0.568 | 25.0% |
| 3 | 97.8% | 9 | 16 | 1 | 0.360 | 0.900 | 0.514 | 66.7% |
| 10 | 100.0% | 15 | 16 | 7 | 0.484 | 0.682 | 0.566 | 25.0% |
| **mean** | **98.8%** | | | | **0.435** | **0.775** | **0.549** | **38.9%** |

**Match%** — node-match recall (CtrMt@50%, predicted valve/instrument nodes matched to GT).
**Cross%** — fraction of GT crossing-separated pairs that appear as a predicted edge (should be
0%; step-4 crossing/connector detection + contraction is wired but not perfect, hence nonzero).

### Gate assessment

**Mean F1 = 0.549 vs. the 0.70 gate — gate NOT met.**

This supersedes the same-day earlier freeze (mean F1 0.545) after a node-set correction (§2)
that turned out to have a small net-positive effect overall (0.545→0.549) despite costing one
sheet-10 TP (§2). None of the fixes to date touch the dominant residual described in §5, which is
why F1 stalls well short of 0.70 regardless of node-set or corridor-logic corrections.

---

## 2. Node Dedupe Correction

**Bug.** `centroid_nms` (step 0b, symbol-prediction NMS) deduped **per class** —
`by_class: dict[cls_id, list[dict]]`. This is correct when two different classes really are
different objects, but several YOLO classes are **mutually-exclusive subtype guesses for the
same kind of physical object**: 5 `instrument_bubble*` classes for one bubble, 14 valve-family
classes (`Symbol_1/3/4/9/10/11/12/13`, `angle_valve`, `valve_handwheel`, `check_valve`,
`control_valve_diaphragm`, etc.) for one valve. Per-class-only NMS never deduped a same-object
pair straddling two of these classes — both survived as separate graph nodes.

**Blast radius measured** (sheets 0/3/10, `scripts/measure_cross_class_duplicates.py`): 45
cross-class prediction pairs with centroids inside the standard NMS radius —

| Family | Close pairs | Disposition |
|---|---|---|
| `instrument_bubble*` (5 classes) | 14 | **Fixed** (prior session) |
| valve (14 classes) | **21** | **Fixed this session** |
| cross-node-type (valve/instrument vs `unknown_fitting`/`flow_arrow`/`tag_rect`) | 10 | **Left open** — needs individual judgment, not a family merge (see below) |

**Valve-family inspection before merging** (`docs/phase4_step4_scope/valve_dedup_crops/`, all 21
pairs cropped and visually checked): ~15 pairs are pure same-glyph subtype confusion (e.g.
`Symbol_10`↔`valve_handwheel` ×5, `Symbol_4`↔`valve_handwheel` ×3 — both boxes tightly bound one
bowtie-valve glyph; consistent with `docs/class_identity/mapping.md` already flagging
`Symbol_4`/`Symbol_12` as "visually ~identical, not resolved"). ~6 pairs
(`control_valve_diaphragm`↔`valve_handwheel` ×4, `Symbol_4`/`Symbol_3`↔`control_valve_diaphragm`)
show one detection spanning a valve body **plus** a separately-drawn "M"-motor-actuator box that
the other detection only partially covers — two distinct drawn sub-elements, but one physical
valve assembly. Both patterns merge to **one** graph node (pipe connections attach at the
bowtie/pinch point either way); the closest-to-threshold case (`Symbol_3`↔`control_valve_diaphragm`,
d=14.8px vs threshold 16.2px) was checked individually and is unambiguously the same assembly.

**Fix**: `scored_family_group_key(cls_id)` (`src/pidetect/graph/erase.py`) groups predictions by
scored supercategory (`"valve"` or `"instrument"`) for NMS purposes; every other class keeps its
own per-class group, unchanged. Survivors get `subtype_conflict=True` +
`suppressed_subtypes=[...]` recorded (visible, not silently resolved) whenever the suppressed
prediction was a **different** class. `assert_no_duplicate_scored_nodes()` runs after node-set
construction in all three production/eval call sites as a regression guard.

**What it inflated**: duplicate valve nodes (avg. ~6/sheet before the fix) each independently
erased, bound endpoints, and competed for short-gap/skeleton edges — manufacturing spurious
graph structure (extra nodes, extra candidate edges) beyond what 106 real valves + 106 real
instruments should produce. Node counts, sheets 0/3/10:

| Sheet | Total preds (before→after) | Valve (before→after) | Instrument (before→after) |
|---|---|---|---|
| 0 | 118→113 | 39→36 | 38→36 |
| 3 | 82→79 | 16→16 | 36→33 |
| 10 | 208→186 | 51→36 | 46→39 |

**Net connectivity effect — mixed, small, net positive overall.** Re-measured all three sheets
after both fixes (§1). Sheet 0 and sheet 3 held steady (sheet 3 has zero valve-family
duplicates, so it's untouched by this fix at all). **Sheet 10 lost one TP**
(`instrumentation62↔valve51`, recall 0.727→0.682) — the OPPOSITE of the bubble fix's effect on
this same sheet last session. Traced directly: two overlapping valve predictions
(`control_valve_diaphragm` conf=0.76, `valve_handwheel` conf=0.30) at the same location near
`valve51`; before the fix both existed as separate nodes and *something* about that configuration
let a short-gap/skeleton edge form to `instrumentation62` that does not form with only the
survivor (`control_valve_diaphragm`) present. This is the same class of subtle
node-position-dependent effect as the bubble fix's sheet-10 *gain* last session, just running the
other way here — not investigated to a root cause (would need the same SR/Sc path-tracing depth
as earlier Phase 4 diagnostics; out of scope for this pass). Net across all three sheets: mean F1
0.545→0.549, still a net improvement over the pre-dedupe baseline despite this one loss — kept the
fix because leaving a confirmed duplicate-node defect in production for the sake of one
incidentally-lucky edge is the wrong tradeoff (CLAUDE.md's "honest metrics always").

**10 cross-node-type pairs left open (known-open, not fixed):** `Symbol_21`↔`flow_arrow` (×3),
`flow_arrow`↔`reducer` (×1), `control_valve_diaphragm`↔`strainer` (×1), `Symbol_21`↔`Symbol_4`
(×1), `Symbol_17`↔`Symbol_5` (×1), `Symbol_9`↔`instrument_bubble` (×1),
`reducer`↔`tag_rectangle_simple` (×1), `Symbol_25`↔`instrument_bubble` (×1). These span
`node_type` boundaries (valve/instrument vs `unknown_fitting`/`flow_arrow`/`tag_rect`, or valve vs
instrument) — exactly the case that must **never** be blindly family-merged (a valve actuator can
legitimately sit right next to an instrument bubble as two real, separate objects). Each needs
individual visual judgment, not a group_key change. Not investigated this session.

---

## 3. Other Fixes Applied Before This Freeze (corridor/short-gap logic, prior session)

1. **Config-wiring correction.** `mask_all_nodes`/`mask_all_nodes_fallback` were defined in
   `configs/phase4.yaml` and documented as the "locked C1" production floor, but no call site
   actually passed them into `run_step3()` — production silently ran `mask_all_nodes=False` the
   entire time. Wiring it naively (clearing every node's dilated bbox from a single shared
   corridor binary, including A's/B's own) cost 5 TPs on sheet 0 by erasing genuine
   connecting-stub ink that lives in a pair's own dilation margin. Fixed by masking **per-pair**:
   every node OTHER than the current (A, B) pair has its dilated bbox cleared, leaving A's/B's own
   margins intact. Net effect: 7 of 8 "third-node-ink" P3b FPs killed at the time, all TPs held.
2. **Tightened bbox-touch bypass.** The short-gap rule unconditionally accepted a pair whose
   original bboxes touch/overlap, with zero ink check. Now requires minimal ink evidence (search
   margin widened 2px→10px after a narrower attempt cost 2 genuine TPs). Killed 2 of 6
   bbox-touch-no-ink FPs; the other 4 survive because a few stray anti-aliasing pixels (circle
   boundary / label serif) are indistinguishable from real ink by pixel count alone. Cost exactly
   1 TP: `sym_106↔sym_110` (sheet 0) — GT scores these twin instrument-tag bubbles as connected
   with **zero visible ink** between them at any search margin tested up to 10px.
3. **Shared-header stub-direction discriminator.** For short-gap pairs where a node has a
   surviving post-erasure skeleton stub, classify its direction into toward/perpendicular/away
   relative to the A→B axis; reject the pair only when neither node's stub points toward the
   other AND at least one points perpendicular (the shared-trunk-tap signature). Nodes with no
   stub at all, or whose stub points roughly opposite (an unrelated far-side port on a multi-port
   symbol), are never treated as evidence against the pair. Final tolerance 55°. Reduced the
   shared-header bucket from 39→32 pairs; the remaining 32 have no surviving stub at all (fully
   consumed by erasure) and are structurally unreachable by this mechanism (§5).

---

## 4. Final FP Composition

| Sheet | Total FP | S4 (topology/crossing) | P3 (no GT path) | OTHER |
|---|---|---|---|---|
| 0 | 27 | 2 (7%) | 23 (85%) | 2 (7%) |
| 3 | 16 | 4 (25%) | 11 (69%) | 1 (6%) |
| 10 | 16 | 2 (12%) | 12 (75%) | 2 (12%) |
| **total** | **59** | **8 (14%)** | **46 (78%)** | **5 (8%)** |

### P3 sub-bucket breakdown

| Sub-bucket | Count | % of P3 | % of total FP | Mechanism |
|---|---|---|---|---|
| **P3b shared-header** | **32** | 70% | 54% | A and B each tap perpendicularly onto a shared trunk parallel to the A→B axis; corridor ink is real but belongs to the trunk, not an A→B line. Dominant residual — see §5. |
| P3c contraction over-bridge | 8 | 17% | 14% | Connector/crossing contraction transitively bridges symbols sharing a header through a chain of (sometimes duplicate-split) junction nodes that GT does not score as directly connected. |
| P3b bbox-touch-no-ink | 4 | 9% | 7% | Original bboxes touch/overlap with zero ink evidence even at a 10px search margin. |
| P3b third-node-ink | 1 | 2% | 2% | Corridor ink attributable to a third detected node's body, surviving the per-pair masking fix (reappeared once post-dedupe; not investigated further). |
| P3ab skeleton-branch (residual) | 1 | 2% | 2% | A short (~35px) spurious skeleton branch directly bridges two unrelated pipe runs. |
| **Total P3** | **46** | 100% | 78% | |

**P3a (dashed/signal-line FPs) = 0%**, unchanged from the prior freeze — this bucket's dominant
mechanism is corridor/geometry (shared-header), not a dashed line, per the visual sample taken in
the earlier session.

---

## 5. Why the Dominant Residual (32 shared-header P3b FPs) Is Unreachable Today

Unchanged from the prior freeze — the node-dedupe correction (§2) did not touch this bucket's
count (32, both before and after). All three discriminator families available to Phase 4 were
tried against it and each was empirically checked:

- **Ink-based** (corridor continuity/perp-reject, `mask_all_nodes`) — cannot distinguish "a
  dedicated A→B line" from "a shared trunk both A and B tap onto," because when the trunk runs
  parallel to the A→B axis the ink is geometrically identical either way.
- **Stub-direction** (shared-header discriminator) — requires a *surviving* skeleton stub to read
  a direction from. 0 of the 32 pairs have any bound endpoint on either node; the connecting pipe
  was fully consumed by `erase_dilation_px`-driven erasure before the skeleton was ever traced.
- **Line-type/dashed-signal classification** (Phase 3, PaddleOCR) — moot since P3a is confirmed
  0%; there is no dashed signal line to classify.

**Two upstream levers could reach this bucket — future work, not undertaken:**

1. **Reduce `erase_dilation_px`** (currently 8px) so a symbol's own short tap-stub survives
   erasure instead of being swallowed by its own dilated bbox, giving the stub-direction
   discriminator evidence to act on. Needs its own TP-safety pass before any config change.
2. **Explicit shared-trunk/header detection** — trace long, roughly-straight skeleton runs
   independently of the short-gap pairwise test, and mark any short-gap candidate whose corridor
   ink coincides with an already-identified trunk segment as a shared-tap, not a connection.

Neither is in scope for the current freeze.

---

## 6. Known Residuals (carried, not fixed)

| Residual | Count | Status |
|---|---|---|
| P3b shared-header | 32 | Dominant; unreachable without an upstream lever (§5) |
| P3c contraction over-bridge | 8 | Root cause identified (duplicate-split junction nodes + naive all-pairs connector contraction across chains); not fixed |
| P3b bbox-touch-no-ink | 4 | Partially fixed (2/6 killed in an earlier pass); remaining 4 indistinguishable from genuine ink by pixel count alone |
| P3b third-node-ink | 1 | Reappeared post-dedupe; not investigated |
| `sym_106↔sym_110` (sheet 0) | 1 TP | Irreducible — GT scores a connection with zero visible ink evidence |
| P3ab skeleton-branch | 1 | Single spurious short skeleton bridge, not investigated |
| `instrumentation62↔valve51` (sheet 10) | 1 TP | Lost by the valve-dedupe fix (§2); not traced to root cause; fix kept anyway on net-improvement + correctness grounds |
| 10 cross-node-type close-prediction pairs (§2) | 10 | Known-open; needs individual visual judgment per pair, not a family-merge |
| OTHER bucket | 5 | Not decomposed |

---

*Phase 4 re-frozen at mean P=0.435 R=0.775 F1=0.549 (three-sheet OPEN100 sample, post node-dedupe
correction). Gate (F1 ≥ 0.70) not met. Next connectivity work items, if resumed: §5's two upstream
levers (dominant residual), or §2's 10 cross-node-type pairs / the `instrumentation62↔valve51`
regression (smaller, more contained investigations).*
