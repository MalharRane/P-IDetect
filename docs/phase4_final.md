# Phase 4 — Final Frozen Record

**Date frozen:** 2026-07-19
**Status:** Connectivity work FROZEN at this state. No further tuning planned until the
upstream levers named in §4 are picked up as new work.

**Config:** `configs/phase4.yaml` as committed alongside this doc —
`mask_all_nodes: true` (per-pair third-party masking, corrected), `mask_all_nodes_fallback: false`,
`min_branch_len_px: 15`, `short_gap_stub_angle_tol_deg: 55` (new in this freeze).
**Sheets evaluated:** 0, 3, 10 (OPEN100)

---

## 1. Frozen Metrics

| Sheet | Match% | TP | FP | FN | P | R | F1 | Cross% |
|---|---|---|---|---|---|---|---|---|
| 0 | 98.6% | 23 | 27 | 8 | 0.460 | 0.742 | 0.568 | 25.0% |
| 3 | 97.8% | 9 | 15 | 1 | 0.375 | 0.900 | 0.529 | 66.7% |
| 10 | 100.0% | 14 | 16 | 8 | 0.467 | 0.636 | 0.538 | 25.0% |
| **mean** | **98.8%** | | | | **0.434** | **0.759** | **0.545** | **38.9%** |

**Match%** — node-match recall (CtrMt@50%, predicted valve/instrument nodes matched to GT).
**Cross%** — fraction of GT crossing-separated pairs that appear as a predicted edge (should be
0%; step-4 crossing/connector detection + contraction is wired but not perfect, hence nonzero).

### Gate assessment

**Mean F1 = 0.545 vs. the 0.70 gate — gate NOT met.**

This is a genuine, measured plateau, not a bug. Three real fixes were applied this session
(config-wiring correction, tightened bbox-touch bypass, shared-header stub-direction
discriminator — see §3) and each one **did** move the number: total FP across the three sheets
fell from 72 (start of session) to 58, entirely by removing false positives, with only one
irreducible TP loss (`sym_106↔sym_110`, sheet 0 — see §3). None of the three fixes touch the
dominant residual described in §2, which is why the F1 gain from 0.509→0.545 stalls well short
of 0.70.

---

## 2. Final FP Composition

| Sheet | Total FP | S4 (topology/crossing) | P3 (no GT path) | OTHER |
|---|---|---|---|---|
| 0 | 27 | 2 (7%) | 23 (85%) | 2 (7%) |
| 3 | 15 | 4 (27%) | 10 (67%) | 1 (7%) |
| 10 | 16 | 2 (12%) | 12 (75%) | 2 (12%) |
| **total** | **58** | **8 (14%)** | **45 (78%)** | **5 (9%)** |

### P3 sub-bucket breakdown (decomposed this session, see prior turns' P3a/P3b/P3c analysis)

| Sub-bucket | Count | % of P3 | % of total FP | Mechanism |
|---|---|---|---|---|
| **P3b shared-header** | **32** | 71% | 55% | A and B each tap perpendicularly onto a shared trunk parallel to the A→B axis; corridor ink is real but belongs to the trunk, not an A→B line. Dominant residual — see §3. |
| P3c contraction over-bridge | 8 | 18% | 14% | Connector/crossing contraction transitively bridges symbols sharing a header through a chain of (sometimes duplicate-split) junction nodes that GT does not score as directly connected. |
| P3b bbox-touch-no-ink | 4 | 9% | 7% | Original bboxes touch/overlap with zero ink evidence even at a 10px search margin; a genuine but purely spatial-adjacency artifact (twin instrument-tag pairs, overlapping icons). |
| P3ab skeleton-branch (residual) | 1 | 2% | 2% | A short (~35px) spurious skeleton branch directly bridges two unrelated pipe runs; not reached by short-gap logic since it's a continuous skan branch, not a short-gap pair. |
| **Total P3** | **45** | 100% | 78% | |

**P3a (dashed/signal-line FPs) = 0%.** Every P3 pair visually sampled this session (9 diverse
pairs across all three sheets, multiple symbol-class combinations) showed the shared-header
tap pattern, not a genuine dashed signal line. This overturns the original `phase4_step3_final.md`
assumption that P3 was dominated by dashed-line FPs — it was actually a corridor/geometry
mislabeling (P3b), now corrected.

---

## 3. What Was Fixed This Session (summary; full detail in conversation history)

1. **Config-wiring correction.** `mask_all_nodes`/`mask_all_nodes_fallback` were defined in
   `configs/phase4.yaml` and documented as the "locked C1" production floor, but no call site
   (`run_phase4_steps03.py`, the eval harness) actually passed them into `run_step3()` — production
   silently ran `mask_all_nodes=False` the entire time. Wiring it naively (clearing every node's
   dilated bbox from a single shared corridor binary, including A's/B's own) cost 5 TPs on sheet 0
   by erasing genuine connecting-stub ink that lives in a pair's own dilation margin. Fixed by
   masking **per-pair**: every node OTHER than the current (A, B) pair has its dilated bbox cleared,
   leaving A's/B's own margins intact. Net effect: 7 of 8 "third-node-ink" P3b FPs killed, all TPs
   held (recall 0.774/0.900/0.636 exactly preserved).
2. **Tightened bbox-touch bypass.** The short-gap rule unconditionally accepted a pair whose
   original bboxes touch/overlap, with zero ink check. Now requires minimal ink evidence (search
   margin widened 2px→10px after the first, narrower attempt cost 2 genuine TPs). Killed 2 of 6
   bbox-touch-no-ink FPs; the other 4 survive because a few stray anti-aliasing pixels (circle
   boundary / label serif) are indistinguishable from real ink by pixel count alone. Cost exactly
   1 TP: `sym_106↔sym_110` (sheet 0) — GT scores these twin instrument-tag bubbles as connected
   with **zero visible ink** between them at any search margin tested up to 10px. This is not a
   corridor-test limitation; there is nothing to detect.
3. **Shared-header stub-direction discriminator.** For short-gap pairs where a node has a
   surviving post-erasure skeleton stub, classify its direction into toward/perpendicular/away
   relative to the A→B axis; reject the pair only when neither node's stub points toward the
   other AND at least one points perpendicular (the shared-trunk-tap signature). Nodes with no
   stub at all, or whose stub points roughly opposite (an unrelated far-side port on a multi-port
   symbol), are never treated as evidence against the pair — this distinction was necessary after
   a naive "any stub not toward = reject" version cost 12 TPs by misreading valves' unrelated
   inlet/outlet stubs as negative evidence. Final tolerance 55° (45° clipped 2 genuine near-boundary
   TPs). Reduced the shared-header bucket from 39→32 pairs; the remaining 32 have **no surviving
   stub at all** (fully consumed by erasure) and are structurally unreachable by this mechanism.

---

## 4. Why the Dominant Residual (32 shared-header P3b FPs) Is Unreachable Today

All three discriminator families available to Phase 4 were tried against this bucket this
session and each was empirically checked against it:

- **Ink-based** (corridor continuity/perp-reject, mask_all_nodes) — cannot distinguish "a
  dedicated A→B line" from "a shared trunk both A and B tap onto," because when the trunk runs
  parallel to the A→B axis the ink is geometrically identical either way.
- **Stub-direction** (Task 3, this session) — requires a *surviving* skeleton stub to read a
  direction from. Checked directly: 0 of the remaining 32 pairs have any bound endpoint on
  either node. The connecting pipe (and each symbol's own tap stub) was fully consumed by
  `erase_dilation_px`-driven erasure before the skeleton was ever traced.
- **Line-type/dashed-signal classification** (Phase 3, PaddleOCR) — moot here since P3a is
  confirmed 0%; there is no dashed signal line to classify. This lever addresses a population
  that, on this evidence, doesn't exist in the current three-sheet sample.

**Two upstream levers could reach this bucket — named here as future work, not undertaken this
session:**

1. **Reduce `erase_dilation_px`** (currently 8px) so that a symbol's own short tap-stub survives
   erasure instead of being swallowed by its own dilated bbox — this would let the stub-direction
   discriminator (already built, §3.3) actually see evidence for these 32 pairs instead of "no
   stub at all." Risk: less erasure margin reintroduces symbol-body skeleton noise elsewhere;
   would need its own TP-safety measurement pass before any config change.
2. **Explicit shared-trunk/header detection** — trace long, roughly-straight skeleton runs
   independently of the short-gap pairwise test, and mark any short-gap candidate whose corridor
   ink coincides with an already-identified trunk segment (rather than a dedicated A-B stub) as
   a shared-tap, not a connection. This is a structurally different mechanism from the pairwise
   corridor/stub tests used so far — a "is this ink part of a longer through-run" question,
   not an "does this ink point at B" question.

Neither is in scope for the current freeze.

---

## 5. Known Residuals (carried, not fixed this session)

| Residual | Count | Status |
|---|---|---|
| P3b shared-header | 32 | Dominant; unreachable without an upstream lever (§4) |
| P3c contraction over-bridge | 8 | Root cause identified (duplicate-split junction nodes + naive all-pairs connector contraction across chains); not fixed this session |
| P3b bbox-touch-no-ink | 4 | Partially fixed (2/6 killed); remaining 4 indistinguishable from genuine ink by pixel count alone |
| `sym_106↔sym_110` (sheet 0) | 1 TP | Irreducible — GT scores a connection with zero visible ink evidence |
| P3ab skeleton-branch | 1 | Single spurious short skeleton bridge, not investigated further |
| OTHER bucket | 5 | Not decomposed this session |

---

*Phase 4 frozen at mean P=0.434 R=0.759 F1=0.545 (three-sheet OPEN100 sample). Gate (F1 ≥ 0.70)
not met. Next connectivity work item, if resumed, should start from §4's two upstream levers.*
