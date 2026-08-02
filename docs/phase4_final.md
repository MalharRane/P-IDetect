# Phase 4 — Final Frozen Record

**Date frozen:** 2026-07-19 (re-frozen same day after the node-dedupe correction, §2)
**Re-frozen:** 2026-08-03 (Fix 2, §4 — erasure-dilation sweep + the already-wired stub-direction
discriminator recover part of the dominant shared-header residual described in the original §5)
**Status:** Connectivity work FROZEN at this state. No further tuning planned until the
remaining items in §7 (dashed-signal-line FPs foremost) are picked up as new work.

**Config:** `configs/phase4.yaml` as committed alongside this doc —
`mask_all_nodes: true` (per-pair third-party masking), `mask_all_nodes_fallback: false`,
`min_branch_len_px: 15`, `short_gap_stub_angle_tol_deg: 55`, **`erase_dilation_px: 3`** (was 8,
lowered by Fix 2 §4).
**Node set:** `src/pidetect/graph/erase.py` — `centroid_nms(..., group_key=scored_family_group_key)`
(valve + instrument_bubble* family-scoped dedup, §2) + `assert_no_duplicate_scored_nodes()`
construction-time regression guard.
**Sheets evaluated:** 0, 3, 10 (OPEN100)

---

## 1. Frozen Metrics

| Sheet | Match% | TP | FP | FN | P | R | F1 | Cross% |
|---|---|---|---|---|---|---|---|---|
| 0 | 98.6% | 23 | 24 | 8 | 0.489 | 0.742 | 0.590 | 25.0% |
| 3 | 97.8% | 9 | 16 | 1 | 0.360 | 0.900 | 0.514 | 66.7% |
| 10 | 100.0% | 16 | 15 | 6 | 0.516 | 0.727 | 0.604 | 25.0% |
| **mean** | **98.8%** | | | | **0.455** | **0.790** | **0.569** | **38.9%** |

**Match%** — node-match recall (CtrMt@50%, predicted valve/instrument nodes matched to GT).
**Cross%** — fraction of GT crossing-separated pairs that appear as a predicted edge (should be
0%; step-4 crossing/connector detection + contraction is wired but not perfect, hence nonzero).

### Gate assessment

**Mean F1 = 0.569 vs. the 0.70 gate — gate NOT met.**

This supersedes the 2026-07-19 freeze (mean F1 0.549) after Fix 2 (§4: `erase_dilation_px` 8→3 +
the already-wired stub-direction discriminator, which together kill part of the dominant
shared-header residual). Net: **+0.020 mean F1, zero TP loss, +1 net TP gain on sheet 10**, all 9
sheet-3 TPs held exactly. This is a genuine correctness improvement (dominant FP cause reduced),
**not gate clearance** — the remaining P3ab(short_gap_ink)/dashed-signal-line residual (§6, §7)
keeps F1 well short of 0.70 regardless.

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
   shared-header bucket from 39→32 pairs at the time; the remaining 32 had no surviving stub at
   all (fully consumed by erasure) and were structurally unreachable by this mechanism — until
   Fix 2 (§4) gave some of them a stub to read.

---

## 4. Fix 2: Erasure-Dilation Sweep + Stub-Direction for Shared-Header FPs (2026-08-03)

**Problem.** At the 2026-07-19 freeze, the shared-header stub-direction discriminator (§3.3) was
correctly wired but starved of evidence: **0 of the 32 shared-header FPs had ANY surviving
post-erasure skeleton stub** on either node, because `erase_dilation_px=8` erased far enough
beyond each symbol's bbox to consume the short connecting tap before the skeleton was ever traced
(original §5 diagnosis). The discriminator logic itself (`_stub_direction_ok`,
`src/pidetect/graph/lines.py`) did not need to change — it just needed something to look at.

**Fix.** Made `erase_dilation_px` a swept parameter and measured, rather than guessed, the value
that lets stubs survive without letting symbol-body glyph residue survive too (the documented
opposing failure mode). `scripts/sweep_erasure_dilation.py` re-runs the full production pipeline
(steps 0–9) at each candidate value on sheets 0/3/10 — nothing else in the config changed,
including `short_gap_stub_angle_tol_deg=55` (already active at every dilation tested).

### Sweep table

`shared_hdr_FP` = FPs in the `P3ab(short_gap_ink)` mechanism bucket specifically (short-gap, real
corridor ink, not third-node-contaminated, not bbox-touch-zero-ink) — confirmed to total exactly
32 at dilation=8, matching the original §3.3/§5 count exactly, which validated the bucket
definition before trusting the sweep. `short_stubs` = skeleton branches < 2×`min_branch_len_px`
with an endpoint bound to a valve/instrument node (the measurable proxy for "symbol-body glyph
residue read as a stub" — the too-little-dilation failure mode named in the brief).

| dil | sheet | TP | FP | FN | P | R | F1 | shared_hdr_FP | short_stubs | connectors | crossings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | 23 | 27 | 8 | 0.460 | 0.742 | 0.568 | 20 | 6 | 63 | 7 |
| 8 | 3 | 9 | 16 | 1 | 0.360 | 0.900 | 0.514 | 5 | 9 | 65 | 16 |
| 8 | 10 | 15 | 16 | 7 | 0.484 | 0.682 | 0.566 | 7 | 11 | 160 | 43 |
| **8** | **mean(pooled)** | 47 | 59 | 16 | 0.443 | 0.746 | 0.556 | **32** | | | |
| 6 | 0 | 23 | 26 | 8 | 0.469 | 0.742 | 0.575 | 19 | 6 | 63 | 7 |
| 6 | 3 | 9 | 15 | 1 | 0.375 | 0.900 | 0.529 | 5 | 6 | 65 | 16 |
| 6 | 10 | 15 | 15 | 7 | 0.500 | 0.682 | 0.577 | 6 | 9 | 162 | 43 |
| **6** | **mean(pooled)** | 47 | 56 | 16 | 0.456 | 0.746 | 0.566 | **30** | | | |
| 4 | 0 | 23 | 26 | 8 | 0.469 | 0.742 | 0.575 | 19 | 7 | 64 | 7 |
| 4 | 3 | 9 | 16 | 1 | 0.360 | 0.900 | 0.514 | 5 | 8 | 65 | 16 |
| 4 | 10 | 16 | 15 | 6 | 0.516 | 0.727 | 0.604 | 6 | 10 | 162 | 43 |
| **4** | **mean(pooled)** | 48 | 57 | 15 | 0.457 | 0.762 | 0.571 | **30** | | | |
| **3** | **0** | **23** | **24** | **8** | **0.489** | **0.742** | **0.590** | **16** | 7 | 66 | 7 |
| **3** | **3** | **9** | **16** | **1** | **0.360** | **0.900** | **0.514** | **5** | 9 | 67 | 18 |
| **3** | **10** | **16** | **15** | **6** | **0.516** | **0.727** | **0.604** | **6** | 9 | 164 | 43 |
| **3** | **mean(pooled)** | **48** | **55** | **15** | **0.466** | **0.762** | **0.578** | **27** | | | |
| 2 | 0 | 23 | 24 | 8 | 0.489 | 0.742 | 0.590 | 17 | 4 | 66 | 7 |
| 2 | 3 | 9 | 17 | 1 | 0.346 | 0.900 | 0.500 | 5 | 10 | 71 | 18 |
| 2 | 10 | 15 | 16 | 7 | 0.484 | 0.682 | 0.566 | 6 | 11 | 165 | 45 |
| **2** | **mean(pooled)** | 47 | 57 | 16 | 0.452 | 0.746 | 0.563 | **28** | | | |

(`mean(pooled)` = TP/FP/FN summed across sheets then P/R/F1 derived, used here to compare sweep
candidates at a glance; §1's headline table uses the doc's usual per-sheet-averaged convention —
mean F1 there, 0.569, is not directly comparable to the 0.578 pooled figure above.)

### TP safety (every dilation tested)

- **Sheet 3's 9 TPs are IDENTICAL across all five dilation values** — zero pair-set change at any
  tested value. Confirmed directly (`TP DIFF vs baseline` in the sweep script's output), not just
  inferred from the count holding.
- Sheet 0: TP set identical to baseline (dilation=8) at every tested value (23, unchanged pairs).
- Sheet 10: dilation=6 shows a **lateral swap** — loses `(29,156)`, gains `(27,172)`, net count
  unchanged (15). Dilations 4 and 3 gain `(27,172)` **without** losing `(29,156)` — a genuine net
  +1 TP, not a swap. Dilation=2 loses the gain entirely, back to the baseline set (15, identical
  pairs). No dilation value in the sweep lost a TP that dilation=8 had and no other value
  recovered.

### Why dilation=3, not the best-F1-looking alternative or the most aggressive value

**Chosen: `erase_dilation_px=3`.** Not simply "the highest F1 row" — the two lower-dilation
neighbors were checked explicitly for the failure mode the brief warned about:

- **Killed the most shared-header FPs net of new noise (7 net across sheets, and the most gross
  P3ab(short_gap_ink) reduction: 32→27)** while **TP-safe** (§ above) and with the **best mean
  F1** among all five candidates — three independent signals agreeing, not one metric
  cherry-picked.
- **Dilation=2 is a measured regression, not a hypothetical one**: fewer shared-header FPs killed
  (28 vs 3's 27 remaining — i.e. worse), sheet 3 gains a **new** FP (16→17, the only dilation
  value where any sheet's FP count goes UP relative to dilation=3), sheet 10's TP gain
  disappears, and mean F1 drops back to 0.563. Sheet 3's connector count also jumps 67→71 at this
  step (largest single jump in the sweep) — the connector-count/crossing-count columns are the
  coarse false-junction proxy, and this is exactly where they move fastest. This is the
  too-little-dilation failure mode named in the brief, caught by measurement rather than assumed.
- **The `short_stubs` proxy is not a clean monotonic signal on its own** (it dips at dilation=6,
  ticks up slightly at 4/3, dips again at 2 on 2 of 3 sheets) — it is too coarse to be the sole
  decision variable, which is exactly why the decision rests on the three converging signals above
  (FP reduction, TP safety, F1) plus the connector/crossing jump at dilation=2, not on this proxy
  alone.

**The win is not "free."** A sheet-0-specific pair-level diff (dilation=3 vs dilation=8 baseline)
shows the 27→24 net FP drop is smaller than the 6 shared-header pairs killed on that sheet alone,
because shrinking dilation changes skeleton/junction *topology*, not just stub visibility: 3 new
`P3c(contracted:connector)` FPs appear (a different, previously-documented residual — contraction
bridging through duplicate-split junction nodes, §6) plus 2 new `P3ab(short_gap_ink)` FPs at
locations that weren't even short-gap candidates at dilation=8 (newly-visible corridor ink that
still reads as a false connection). Net effect stays positive (8 gone, 5 new = -3 FP on sheet 0
alone), but "shared-header FPs killed" and "net FP reduction" are not the same number, and both
are reported above rather than only the more flattering one.

**FP composition, dilation=3** (`scripts/decompose_p3.py`, run against the new config):

| Sub-bucket | Count | Was (dilation=8) | Δ |
|---|---|---|---|
| P3ab(short_gap_ink) "shared-header" | 27 | 32 | **-5** |
| P3c(contracted:connector) | 11 | 8 | **+3** (new topology-driven FPs, see above) |
| P3b(bbox_touch_no_ink) | 4 | 4 | 0 |
| P3b(third_node_ink) | 0 | 1 | -1 |
| P3ab(skeleton_branch) | 0 | 1 | -1 |
| **Total P3** | **42** | **46** | **-4** |

---

## 5. Final FP Composition (current freeze)

| Sheet | Total FP | S4 (topology/crossing) | P3 (no GT path) | OTHER |
|---|---|---|---|---|
| 0 | 24 | 2 (8%) | 21 (88%) | 1 (4%) |
| 3 | 16 | 4 (25%) | 11 (69%) | 1 (6%) |
| 10 | 15 | 2 (13%) | 10 (67%) | 3 (20%) |
| **total** | **55** | **8 (15%)** | **42 (76%)** | **5 (9%)** |

### P3 sub-bucket breakdown

| Sub-bucket | Count | % of P3 | % of total FP | Mechanism |
|---|---|---|---|---|
| **P3b shared-header** (`P3ab(short_gap_ink)`) | **27** | 64% | 49% | A and B each tap perpendicularly onto a shared trunk parallel to the A→B axis; corridor ink is real but belongs to the trunk, not an A→B line. Reduced 32→27 by Fix 2 (§4); dominant residual — see §6. |
| P3c contraction over-bridge | 11 | 26% | 20% | Connector/crossing contraction transitively bridges symbols sharing a header through a chain of (sometimes duplicate-split) junction nodes that GT does not score as directly connected. Grew 8→11 as a side effect of Fix 2's dilation change (§4) — same root cause as before, more surface area. |
| P3b bbox-touch-no-ink | 4 | 10% | 7% | Original bboxes touch/overlap with zero ink evidence even at a 10px search margin. Unchanged by Fix 2. |
| P3b third-node-ink | 0 | 0% | 0% | Gone at dilation=3 (was 1). |
| P3ab skeleton-branch (residual) | 0 | 0% | 0% | Gone at dilation=3 (was 1). |
| **Total P3** | **42** | 100% | 76% | |

**P3a (dashed/signal-line FPs) = 0%**, unchanged — this bucket's dominant mechanism remains
corridor/geometry (shared-header), not a dashed line.

---

## 6. Why the Remaining Shared-Header Residual (27 FPs) Is Still Not Fully Reachable

Fix 2 (§4) recovered 5 of the original 32 shared-header FPs by giving the stub-direction
discriminator evidence at 3px dilation, but 27 remain. Re-checked the same three discriminator
families against the residual:

- **Ink-based** (corridor continuity/perp-reject, `mask_all_nodes`) — still cannot distinguish "a
  dedicated A→B line" from "a shared trunk both A and B tap onto" when the trunk runs parallel to
  the A→B axis; unchanged by the dilation fix (this mechanism never looked at stubs).
- **Stub-direction** (shared-header discriminator) — now has evidence for *some* pairs (5 killed),
  but most of the 27 remaining still show **no surviving stub even at 3px** — the connecting tap
  is shorter than the erasure footprint can preserve without also risking symbol-body glyph
  residue (§4's dilation=2 regression is the direct evidence that pushing further starts costing
  more than it recovers). This is a harder floor than "erase less," not a wiring gap.
- **Line-type/dashed-signal classification** (Phase 3, PaddleOCR) — still moot; P3a is still 0%,
  there is no dashed signal line to classify.

**Upstream levers still not undertaken:**

1. **Explicit shared-trunk/header detection** — trace long, roughly-straight skeleton runs
   independently of the short-gap pairwise test, and mark any short-gap candidate whose corridor
   ink coincides with an already-identified trunk segment as a shared-tap, not a connection. This
   does not depend on stub survival at all, so it is not subject to the dilation tradeoff — the
   most promising untried lever for the residual 27.
2. **P3c(contracted:connector) growth (8→11, §4)** — worth a dedicated investigation now that
   Fix 2 has made it larger and more visible; same root cause as the pre-existing 8 (duplicate-split
   junction nodes + naive all-pairs connector contraction across chains).

Neither is in scope for the current freeze.

---

## 7. Known Residuals (carried, not fixed)

| Residual | Count | Status |
|---|---|---|
| P3b shared-header | 27 (was 32) | Dominant; partially recovered by Fix 2 (§4); remainder needs the explicit shared-trunk lever (§6) |
| P3c contraction over-bridge | 11 (was 8) | Root cause identified (duplicate-split junction nodes + naive all-pairs connector contraction across chains); grew as a Fix 2 side effect; not fixed |
| P3b bbox-touch-no-ink | 4 | Partially fixed (2/6 killed in an earlier pass); remaining 4 indistinguishable from genuine ink by pixel count alone |
| `sym_106↔sym_110` (sheet 0) | 1 TP | Irreducible — GT scores a connection with zero visible ink evidence |
| `instrumentation62↔valve51` (sheet 10) | 1 TP | Lost by the valve-dedupe fix (§2); not traced to root cause; fix kept anyway on net-improvement + correctness grounds |
| 10 cross-node-type close-prediction pairs (§2) | 10 | Known-open; needs individual visual judgment per pair, not a family-merge |
| OTHER bucket | 5 | Not decomposed |

---

*Phase 4 re-frozen at mean P=0.455 R=0.790 F1=0.569 (three-sheet OPEN100 sample, post Fix 2
erasure-dilation sweep). Gate (F1 ≥ 0.70) not met — dominant FP cause reduced, F1 up, gate not
cleared. Next connectivity work items, if resumed: §6's explicit shared-trunk/header detection
(the residual 27 shared-header FPs, not reachable by further dilation tuning per §4's dilation=2
regression), or the newly-grown P3c(contracted:connector) bucket (§6.2).*
