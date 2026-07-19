# Phase 4 Step 4 Design — Crossing/Connector Detection + Short-Gap Veto

**Date:** 2026-07-16  
**Basis:** Step-3 three-sheet FP breakdown; Decision-1 geometric approach in `docs/phase4_design.md §1`  
**Scope:** Design only. No implementation here.

---

## 1. What Step 3 Left Behind

Step 3 reaches mean F1=0.51 (P=0.40, R=0.77) across three sheets. The 62 surviving FPs
across all sheets split into two categories that step-3 corridor tuning cannot close:

| Category | Count (all sheets) | Root cause |
|---|---:|---|
| **GT-reach FPs** — path exists in raw GT graph, but predicted A→B edge is wrong | ~42 | Short-gap edge fired for crossing-topology (bucket ii) or symbol-shortcut (bucket i) path |
| **GT-disconnected FPs** — no GT path at all | ~20 | Neighbour pipe runs parallel to A→B in the gap; corridor check cannot distinguish it from a connecting pipe without junction context |

Step 4's job: eliminate the GT-reach FPs (the majority) and as many GT-disconnected FPs as
junction context makes possible, without dropping recall below 0.75.

---

## 2. Crossing/Connector Detection — Decision-1 Confirmed

**Decision 1 (geometric skeleton approach) from `docs/phase4_design.md §1` is confirmed and
already implemented in `src/pidetect/graph/junction.py` via `run_step4`.** Nothing in the
step-3 analysis argues for changing the algorithm; the issue is the pipeline wiring downstream
of crossing detection, not the detection itself.

### 2a. What the algorithm produces

`run_step4` operates on the post-erasure skeleton. For every candidate site (degree≥3 skeleton
node or cluster of branch endpoints within radius R=10px), it:

1. Computes branch directions from the site centroid.
2. Pairs collinear branches (angular deviation from 180° within 30°).
3. Gap-tests each collinear pair: samples pixel continuity along the straight line between
   the last pixel of one branch and the first pixel of the opposing branch.
   - Gap ≥ G_min (20px) and ≤ G_max (65px) → **crossing** (bridge-gap convention).
   - No gap → **connector** (lines physically join).

Confirmed OPEN100 bridge-gap range: 31–60px. G_min=20, G_max=65 already accommodates both
orientations with margin.

### 2b. Crossing as a graph object

A detected crossing produces a `CrossingNode` with:

- **`cx`, `cy`** — centroid of the endpoint cluster, in full-sheet pixel coords (same
  space as symbol centroids). This is the location used for the veto checks below.
- **`pass_through` pairs** — two (node_A, node_B) pairs: the two collinear branches
  that represent the two pipes passing over each other. Only these pairs get edges in the
  contracted graph; no cross-pairing is added.
- **`branch_ids`** — the four (or two, if degree-2 facing-stub form) skeleton branch IDs
  bound to this crossing. Used during graph construction to route branch-to-node edges.

A detected connector produces a `ConnectorNode` with `cx`, `cy` and the set of bound branch
IDs. Connector nodes are contracted out (all-pairs edges among their neighbours, then remove
the node); no special pass-through logic is needed.

---

## 3. The Architectural Gap — Why Short-Gap Edges Currently Bypass Crossing Detection

This is the core issue.

Short-gap edges represent pipes whose connecting segment was **fully consumed by the dilated
bbox erasure**. They are detected in step 3 via the pre-erasure binary corridor check
(`find_short_gap_pairs`) and inserted into the graph in step 5 as **direct A→B edges**,
not routed through any intermediate node.

When step 4 runs crossing detection on the **post-erasure skeleton**, it finds crossings on
pipe segments that survived erasure — but the short-gap region (the erased pipe between A and
B) has no skeleton pixels. Therefore:

- A crossing that lies ON the A→B pipe but WITHIN the erased zone: invisible to step 4.
- A crossing that lies ON the A→B pipe but OUTSIDE the erased zone: visible and detected.
  Its centroid is known precisely from the skeleton endpoint cluster.

When step 6 contracts the graph, it removes crossing **nodes** from the graph with
pass-through semantics. But the short-gap edge (A, B) was never routed through the
crossing node — it was inserted as a direct edge in step 5, bypassing the crossing node
entirely. Contraction leaves it untouched. This is why the 42 GT-reach FPs survive even
though step 4 already runs and step 6 already contracts crossings.

**The fix is not in the detection or contraction algorithm. It is in the wiring: crossing
knowledge must flow from step 4 back into the short-gap firing rule before step 5 inserts
the edges.**

---

## 4. The Short-Gap Crossing Veto

### 4a. The rule

After `run_step4` produces the list of detected crossings, filter `s3.short_gap_pairs` to
remove any pair (A, B) for which a detected crossing node lies on the A→B segment:

> Veto pair (A, B) if there exists a detected crossing C such that:
> - C's centroid projects onto segment A→B at parameter t ∈ [0.10, 0.90], AND
> - The perpendicular distance from C's centroid to line A→B ≤ `crossing_veto_tol_px`
>   (suggested initial value: 15px — tighter than the symbol interpose tolerance of 20px
>   because crossing centroids are geometrically precise skeleton nodes, not noisy YOLO
>   bounding-box centroids).

This is structurally identical to `_has_no_interposing_node` (already implemented in
`lines.py`) applied to crossing nodes instead of symbol nodes. The t-range is slightly
wider (0.10–0.90 vs 0.15–0.85) to account for crossings near the symbol edges.

The vetoed pairs are dropped from `short_gap_pairs` before step 5 adds them to the graph.
Unvetoed pairs are still inserted as before.

### 4b. Data flow

```
Step 3:  run_step3(..., binary_pre_erase, corridor_params)
         → s3.short_gap_pairs  (corridor-filtered candidate pairs)

Step 4:  run_step4(s3.skeleton, s3.branches, s3.endpoints, ...)
         → s4.crossings  (list of CrossingNode with .cx, .cy)

Step 4b: [NEW] veto_short_gap_at_crossings(
             s3.short_gap_pairs, s4.crossings, all_nodes,
             tol_px=crossing_veto_tol_px, t_range=(0.10, 0.90)
         )
         → s3_vetoed.short_gap_pairs  (crossing-filtered pairs)

Step 5:  run_step5(all_nodes, s3_vetoed.off_page_nodes, s3_vetoed, s4)
         → pre-contracted graph  (short-gap edges added from s3_vetoed.short_gap_pairs)

Step 6:  run_step6(s5)
         → contracted graph  (crossing nodes contracted with pass-through semantics;
                               crossing-vetoed pairs are already gone, no duplication)
```

`veto_short_gap_at_crossings` is a pure function: `(pairs, crossings, nodes, tol, t_range)
→ filtered_pairs`. It does not modify any step result in place; the caller creates
`s3_vetoed = replace(s3, short_gap_pairs=filtered_pairs)` before passing to step 5.

The new config param `crossing_veto_tol_px` goes in `configs/phase4.yaml`. Suggested
initial value: 15. A value too small (< 10px) misses crossings that are slightly
off-axis from the A→B centreline; too large (> 25px) risks vetoing pairs that
have a coincidentally-nearby crossing on a different pipe.

### 4c. Coverage limitations

The veto can only fire for crossings that are **outside the erased zone**. A crossing
embedded within the dilated-bbox overlap region is invisible to the post-erasure skeleton
and therefore undetectable by step 4. For typical OPEN100 P&ID layouts, crossings occur
on the pipe run, not inside symbol body zones, so this case is expected to be rare. Cases
where it occurs will remain as FPs regardless of veto tolerance tuning; they are inherent
to the erasure-then-skeleton pipeline order.

---

## 5. Intermediate-Symbol Shortcut Suppression

### 5a. Why the geometric interpose check is insufficient

`_has_no_interposing_node` (step 3) already blocks pairs where a third symbol C's centroid
projects onto A→B at t∈[0.15, 0.85] within 20px perpendicularly. Bucket-i FPs that
survive are those where C is slightly off the A→B centreline (perp > 20px or t outside
range) even though the real pipe route IS A→C→B through C.

After the crossing veto (§4) and graph construction (step 5), the pre-contracted graph
contains both the short-gap edge (A,B) AND the skeleton-traced edges A→C and C→B (if C
is connected to A and B via skeleton branches). This provides a graph-level test that the
geometric check cannot: the connectivity of C to both A and B is visible in the graph.

### 5b. The graph-corroborated interpose check

Applied as a pre-contraction pass in step 6, after the full pre-contracted graph exists:

> Suppress short-gap edge (A, B) if there exists a scored symbol node C such that:
> 1. C is adjacent to both A and B in the pre-contracted graph (i.e., a path A–C–B of
>    length exactly 2 exists through C), AND
> 2. C's centroid projects onto segment A→B at t ∈ [0.10, 0.90] with perpendicular
>    distance ≤ `symbol_shortcut_perp_px` (suggested 30px — looser than the interpose
>    filter to catch off-axis cases that the geometric check missed), AND
> 3. The pair (A, B) is a **short-gap pair** (edge attribute `short_gap=True`). Do NOT
>    suppress skeleton-traced edges (which route through the skeleton naturally and do
>    not create shortcuts independently).

The path-length-2 condition (exactly A–C–B, not longer paths) prevents false suppression
when A and B are in the same connected component via a distant route that does not involve
a geometrically interposing symbol.

The t/perp condition (condition 2) prevents suppression when C just happens to be
adjacent to both A and B for an unrelated reason (e.g., a T-junction where C branches
to both A and B from a perpendicular direction).

Condition 3 limits the pass to short-gap edges only. Skeleton-traced edges (which
emerge from the branch→node binding) are already spatially correct; suppressing them
would break valid connectivity.

### 5c. Ordering within step 6

Run the graph-corroborated interpose check BEFORE connector/crossing contraction:

```
Step 6:
  6a. [NEW] Graph-corroborated interpose pass — suppress short-gap (A,B) edges
            where a scored symbol C lies at graph-distance 1 from both A and B
            and is geometrically between them.
  6b. Delete crossing-to-crossing edges (existing).
  6c. Contract all connector nodes (existing).
  6d. Contract all crossing nodes with pass-through semantics (existing).
  → Contracted symbol-to-symbol graph.
```

Running 6a before 6b–6d matters: after contraction, the intermediate node C may have
been merged into its neighbours, erasing the evidence needed for the length-2 path check.
Connectors are contracted; if C is itself a connector, it won't be in the graph by the
time 6a would run post-contraction. Run 6a on the pre-contracted graph where all nodes
including connectors are present. Scored symbols (valve, instrument, unknown_fitting)
are never contracted, so the distinction is clear.

New config param: `symbol_shortcut_perp_px` in `configs/phase4.yaml`. Suggested initial
value: 30px. This is wider than the existing 20px interpose tolerance to catch the
off-axis bucket-i cases that the geometric check already missed.

---

## 6. Expected Impact Per FP Bucket

| Bucket | Sheet-3 count | Expected step-4 action |
|---|---:|---|
| **ii — Crossing topology** (GT path cut by crossing contraction) | ~7 | **Crossing veto (§4)** removes most. Residual: crossings inside the erased zone. |
| **i — Symbol shortcut** (GT path A→C→B, we fire A→B) | ~7 remaining after corridor | **Graph-corroborated interpose (§5)** catches those where C is connected to both A and B. Residual: cases where C is not in the pre-contracted graph (C is a vessel or off-page node). |
| **iii — GT-disconnected** (no GT path at all) | ~3 SG + contraction ~6 = 9 total | **Partially** addressed: junction knowledge may allow the GT-disconnected SG pairs to be resolved if their gap region contains a detected junction/crossing that redirects their pipe. Otherwise structural residual. |
| **iv — Same-run-adjacent** (endpoints on same pipe, not GT-adjacent) | ~2 | Likely caught by the crossing veto if a crossing separates them; otherwise structural. |

Rough quantitative expectation (sheet 3 as example, from 20 FP baseline):
- Crossing veto removes ~5–7 of the 11 GT-reach FPs (the crossing-topology sub-bucket).
- Graph-corroborated interpose removes ~2–4 more (off-axis symbol shortcuts).
- GT-disconnected: 3 likely persist (no step-4 fix for ambient parallel-pipe cases).
- 6 contraction-origin FPs: step 4 crossing detection should reduce these (better crossing
  classification means fewer spurious contraction paths through non-crossings).

Combined projection, sheet 3: FP ~20 → ~8–10, with TP=9 held. P: 0.310 → ~0.47–0.53.

Across all three sheets: FP ~71 → ~35–45, P mean ~0.40 → ~0.55–0.65. Recall expected
to hold because both vetoes only remove edges — they never suppress a correctly predicted
TP edge, provided crossing detection and the graph structure are correct.

---

## 7. Success Metric

**Gate for step 4 complete:**

| Metric | Sheet 0 | Sheet 3 | Sheet 10 | Mean |
|---|---|---|---|---|
| Edge precision | ≥ 0.60 | ≥ 0.55 | ≥ 0.60 | ≥ 0.60 |
| Edge recall | ≥ 0.75 | ≥ 0.85 | ≥ 0.60 | ≥ 0.75 |
| Edge F1 | ≥ 0.67 | ≥ 0.67 | ≥ 0.62 | ≥ 0.67 |

These targets are conservative: they leave room for imperfect crossing detection
(missed crossings → crossing veto does not fire → some bucket-ii FPs survive).
Achieve 0.70+ mean F1 → proceed to step 5 (text / OCR binding).

Additionally report crossing non-edge rate (defined in `docs/phase4_design.md §4`).
Target ≤ 30% as a minimum (better than a naive connector-only approach); ultimate
goal ≤ 20% as specified in the phase gate.

---

## 8. Residual Recall Gap — Not a Step-4 Issue

Sheet 10 has FN=8 (recall 0.636) even BEFORE step 4. The missing connections are
valve→instrument pairs where the connecting pipe was fully erased with no surviving
corridor ink in the pre-erasure binary — `_has_aligned_corridor_path` returns False and
no short-gap edge fires.

Root cause: for some valve→instrument pairs on sheet 10, the dilation margin (8px) is
large enough to consume a short pipe segment entirely, AND the pre-erasure image shows no
clear aligned ink (the pipe is too thin or the instrument glyph's own ink fills the gap
ambiguously).

**Step 4 does not fix this.** Step 4 cannot recover a pipe that was erased from the
image and left no detectable skeleton stub. The fix would require either:
- A smaller dilation margin for specific symbol classes (instrument_bubble), with the
  risk of re-introducing skeleton noise from symbol-body pixels.
- A step-3 enhancement: for an instrument bubble with zero detected pipe stubs, fall
  back to the nearest valve within `short_gap_max_px` that has a stub pointing toward it.

This is an erasure-layer issue. It should not be misattributed to step-4 crossing
detection when sheet-10 recall is evaluated. Track it as a separate open item for after
step 4 is stable.

---

## 9. Implementation Checklist (for reference when coding begins)

In order of dependency:

1. **`veto_short_gap_at_crossings` function** (new) — pure function, probably lives in
   `src/pidetect/graph/lines.py` or `src/pidetect/graph/junction.py`.
   Inputs: `pairs: list[tuple[int,int]]`, `crossings: list[CrossingNode]`,
   `nodes: list[SymbolNode]`, `tol_px: float`, `t_range: tuple[float,float]`.
   Output: filtered `pairs`.

2. **Call site in `scripts/run_phase4_steps03.py`** — between `run_step4` and `run_step5`:
   compute `filtered_pairs`, create `s3_vetoed`, pass to `run_step5`.

3. **New config param** `crossing_veto_tol_px: 15` in `configs/phase4.yaml`.

4. **Graph-corroborated interpose pass in `run_step6`** (new, `src/pidetect/graph/build.py`)
   — before the existing crossing-to-crossing edge deletion. Iterate short-gap edges,
   apply the path-length-2 + geometric condition, suppress where both conditions hold.

5. **New config param** `symbol_shortcut_perp_px: 30` in `configs/phase4.yaml`.

6. **Evaluate on all three sheets** against the §7 gates. If mean P still below 0.60,
   diagnose which FP bucket dominates the residual and tune `crossing_veto_tol_px` or
   the t-range. Do not retune per-sheet; report honest three-sheet numbers.

7. **Report crossing non-edge rate** (already computed by `run_step9`) alongside P/R/F1.
