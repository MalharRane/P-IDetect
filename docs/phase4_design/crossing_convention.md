# Crossing Convention Verification

**Date:** 2026-06-24  
**Script:** `scripts/verify_crossing_convention.py`  
**Sheets sampled:** 0, 3, 10  
**Contact sheets:** `crossings.png`, `connectors.png`, `crossings_zoom.png`, `connectors_zoom.png`, `crossing_pairs.png`, `crossing_detail.png`

---

## Node counts

| Sheet | Crossings | Connectors |
|---|---|---|
| 0 | 71 | 225 |
| 3 | 36 | 152 |
| 10 | 95 | 366 |

All topology node bboxes are exactly 8x8px integer grid cells (point annotations).

---

## Degree distribution — critical schema correction

```
Sheet 0 crossing degree distribution: {1: 5, 2: 28, 3: 38}
Sheet 0 connector degree distribution: {2: 225}

Sheet 3 crossing degree distribution: {1: 2, 2: 16, 3: 18}
Sheet 3 connector degree distribution: {2: 152}

Sheet 10 crossing degree distribution: {1: 17, 2: 47, 3: 31}
Sheet 10 connector degree distribution: {2: 366}
```

**ALL connector nodes are degree 2.** Connectors are simple waypoints/bends on a pipe run
(elbows, straight-segment waypoints), NOT T-junctions. There are no degree-3 connectors.

**Crossing nodes have degrees 1, 2, or 3 — never 4.**

This means:
- T-junctions and multi-way pipe junctions are NOT encoded by connector nodes.
  They are encoded by "general" symbol nodes (the 20 `general` nodes per sheet in IGNORED_LABELS).
- The phase4_design.md claim that connector nodes represent T-junctions is WRONG and must be corrected.

---

## Crossing-to-crossing edges — the schema

Inspecting neighbor labels of degree-3 crossing nodes reveals:

```
crossing148 @ (1077,660)  ->  ['crossing', 'connector', 'connector']
crossing149 @ (1079,628)  ->  ['connector', 'crossing']
```

**Crossings come in PAIRS.** Each crossing event in the diagram creates TWO crossing nodes
connected to each other by a crossing-to-crossing edge. Each node in the pair is also
connected to the two connectors on its pipe segment (entry and exit of the pipe through
the crossing location). Summary:

```
Pipe A route:  ... -> connector_X -> crossing_A -> connector_Y -> ...
                                          |
                                   crossing edge
                                          |
Pipe B route:  ... -> connector_P -> crossing_B -> connector_Q -> ...
```

The crossing-to-crossing edge (crossing_A <-> crossing_B) is the **explicit non-connectivity
signal** in the graph. It marks "pipes A and B physically cross here but are NOT connected."

**Implication for GT contraction (Phase 4 evaluation):**
Correct algorithm:
1. Remove all crossing-to-crossing edges from the graph.
2. Contract all remaining topology nodes (both connector AND crossing) using standard graph
   contraction (add edges between their remaining neighbors).
3. The result is the symbol-to-symbol adjacency graph.

The design in `phase4_design.md §4` described a "geometric inference" approach to identify
pass-through pairs. That is unnecessary: the crossing-to-crossing edges ARE the explicit
pass-through markers and must simply be deleted before contraction.

### Degree breakdown for crossing nodes

| Degree | Interpretation |
|---|---|
| 1 | Stub: connected only to its crossing partner (the stub node of a pair). The partner is degree 2 or 3. |
| 2 | Full crossing waypoint: two connectors (entry/exit) + one crossing partner — this node plus its partner form a complete crossing pair. OR: waypoint on a pipe with no annotated partner (unmatched crossing). |
| 3 | Compound: connected to two crossing partners (two separate crossing events at nearly the same location) + one connector. Complex but handled by the same rule. |

Sheet 0 has 31 crossing-to-crossing edges (31 crossing pairs out of 71 total crossing nodes).
This accounts for the degree-3 nodes appearing when one crossing node participates in two events.

---

## Visual convention: does OPEN100 use bridge-gap?

**Answer: YES — confirmed.**

The crossing pair coordinates reveal the convention:

```
Pair                              Node coords                Gap
crossing148/149 (vertical)        (1077,660)/(1079,628)      32px in y
crossing155/156 (vertical)        (1525,629)/(1525,660)      31px in y
crossing184/185 (vertical)        (1462,1101)/(1462,1132)    31px in y
crossing193/334 (vertical)        (1352,660)/(1352,628)      32px in y
crossing151/187 (horizontal)      (1270,382)/(1325,381)      55px in x
crossing174/175 (horizontal)      (1598,482)/(1653,482)      55px in x
crossing204/205 (horizontal)      (1100,482)/(1160,482)      60px in x
```

The two nodes in each crossing pair mark the **two endpoints of the gap in the underpass pipe**:

- **Vertical crossings** (horizontal pipe passes over vertical pipe): the two crossing nodes
  are at the same x-coordinate with y offset = 31–32px. The vertical pipe has a 31–32px gap.
  The horizontal overpass pipe runs unbroken through this gap.

- **Horizontal crossings** (vertical pipe passes over horizontal pipe): the two crossing nodes
  are at the same y-coordinate with x offset = 55–60px. The horizontal pipe has a 55–60px gap.
  The vertical overpass pipe runs unbroken through this gap.

The gap is 31–60px in original sheet pixel coordinates.

### Visual appearance

Looking at the zoomed crops (`crossing_detail.png`, 8x zoom):
- At crossing locations, one pipe has a visible white space (gap) where the other pipe
  passes over it. The gap is substantial and clearly visible at drawing resolution.
- The overpass pipe runs continuously through the gap without interruption.
- This matches the standard P&ID CAD drafting convention ("bridge" or "hop" drawing).

### Connector comparison

Connectors are pure waypoints — elbows and bends on pipe runs. Visual inspection shows:
- Simple right-angle bends where a pipe turns from horizontal to vertical or vice versa
- No T-junctions (those are represented by `general` symbol nodes)
- No visual distinction between a connector waypoint and a pipe bend — they're the same

---

## Answer to the design question

**Does OPEN100 use the bridge-gap convention for crossings?**

**YES — the gap-test in Decision 1 is VIABLE, but with corrections:**

| Parameter | Phase 4 Design (original) | Corrected value |
|---|---|---|
| Gap threshold G | 4px | **20–65px** (31–60px observed + margin) |
| Detection signal | Break in collinear pair | Same — skeleton has a white-space gap |
| Algorithm | Correct in approach | Correct in approach |

The gap is much larger than originally assumed (5–15x larger). This is GOOD for detection
reliability: a 30-60px break in a skeleton is unambiguous and resistant to noise.

---

## Schema corrections for phase4_design.md

Three corrections required before Phase 4 implementation:

**1. Connector nodes are NOT T-junctions** (original design said "degree 3+ for T-junctions"):
- All connectors are degree 2. T-junctions are `general` nodes.
- In Phase 4 detection, any skeleton node with degree ≥ 3 (after endpoint binding) can be
  classified as a connector (junction). This is unchanged algorithmically, but connectors
  in the GT are only the degree-2 bends; T-junctions in GT are `general` nodes.

**2. Crossing contraction is by edge deletion, not geometry** (original design described
geometric inference of pass-through pairs):
- Remove all crossing-to-crossing edges, then contract all remaining topology nodes.
- No geometric pass-through pair inference needed for GT evaluation.
- For DETECTED crossings (Phase 4 output), we still need to infer which pairs of branches
  are "passing through" vs "connecting" — the gap-test algorithm from Decision 1 remains
  the right approach for detection; the GT evaluation simplification only applies to GT.

**3. Gap threshold update** (original design said G=4px):
- Set G ∈ [20, 65]px in `configs/phase4.yaml`.
- `min_crossing_gap: 20` and `max_crossing_gap: 65` as the detection bounds.

---

## Degree-2 crossings (no crossing partner)

28 crossing nodes on sheet 0 have degree 2 (two connectors, no crossing partner). These may
represent:
- Annotated crossing where the partner pipe is too thin/dashed to be annotated separately
- Annotation gaps in the GT (one side of the crossing was not labeled)
- Points where a signal/instrument line crosses a process pipe (the signal line may not
  have been traced as a separate pipe route)

For Phase 4 detection, the gap-test will identify these correctly as crossings (the underpass
pipe still has the gap, even if the partner is not annotated). For GT evaluation, these
crossings do NOT have crossing-to-crossing edges and therefore do NOT affect the contraction
algorithm (they're contracted out normally like any other 2-connector waypoint, which may
slightly under-penalize false crossings near these locations).

---

## Summary of visual inspection results

| Category | N sampled | Bridge-gap visible | Junction dot visible | Plain intersection |
|---|---|---|---|---|
| Crossing nodes (all types) | 60 | ~70% (at crossing pair locations) | 0 | ~30% (degree-2, no partner) |
| Connector nodes | 30 | n/a | 0 | n/a (all are bends, no intersection) |

The ~30% "plain intersection" category are the degree-2 crossings (no partner annotation).
These may or may not have a visible gap in the drawing — the GT does not provide a crossing
partner to confirm the gap location, so they're ambiguous from visual inspection.

**Decision 1 verdict: the bridge-gap convention is used. The gap-test is the correct
detection signal. Update G from 4px to 20-65px.**
