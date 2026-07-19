# Phase 4 Design — Connectivity Graph Extraction

**Date:** 2026-06-24  
**Status:** Steps 0–3 implemented and evaluated. Step 4 is next.  
**Basis:** PID2Graph two-tier GraphML schema — see `docs/connectivity_readiness.md §5`

---

## Context

Phase 1 gives us symbol nodes with strong recall (valve 96.8%, instrument 100%,
arrow 56% — see `docs/phase1_8_final.md §D`). Phase 4 turns those detected symbols
plus the pipe-line image into a connectivity graph evaluated against PID2Graph's
GraphML ground truth.

The target format is well-understood from `docs/connectivity_readiness.md` and
`docs/phase4_design/crossing_convention.md` (verified schema):

- Two-tier nodes: symbol nodes (float bbox) + topology nodes (connector, crossing)
- Undirected; all edges carry `edge_label = "solid"`
- Sheet 0: 225 connector nodes (ALL degree 2 — bends/waypoints) + 71 crossing nodes
  (degrees 1–3, coming in pairs connected by crossing-to-crossing edges) + 162 symbol-type nodes
- Connector contraction: delete crossing-to-crossing edges first, then contract all topology
  nodes and `general` (T-junction) nodes to recover symbol-to-symbol edges
- Crossings are NON-connections; the crossing-to-crossing edge is the explicit non-connectivity signal
- Bridge-gap convention confirmed: underpass pipe has a 31–60px gap in the drawing

Three constraints from the readiness check feed directly into pipeline design:

1. **21 title-block FPs** -- suppress with a process-area ROI mask before graph construction
2. **22 tile-seam duplicate predictions** -- centroid-distance NMS dedupe before erasure
3. **104 "FP other" in-diagram predictions = real unscored symbols** -- keep as `unknown_fitting` nodes (dropping them breaks pipe reachability)

---

## Decision 1: Connector/Crossing Detection

### The fork

**Option A -- Object detection:** Train or rule-classify line intersections as
connector vs crossing objects, supervised by PID2Graph junction bboxes.

**Option B -- Geometric inference from skeleton:** Trace lines on the symbol-erased
image; classify each skeleton intersection by collinearity and gap analysis.

### Decision: Option B (geometric skeleton)

**Why Option A fails here:**

A connector is not a distinct glyph. Its visual appearance is "lines meeting at a
point" -- indistinguishable from surrounding pipe pixels at any scale. A detector
trained on connector crops would be learning local line density, not a meaningful
visual feature. The GT bboxes for connectors are point-sized integer rects; at these
scales there is no reliable receptive field. Object detection is appropriate for
symbols (valves, instruments) because they have distinctive shapes. Topological
annotations (connector, crossing) are structural properties of the LINE NETWORK, not
objects with visual signatures.

**Why Option B is correct:**

The connector/crossing distinction IS a geometric property of the skeleton:

- **Connector** = a bend or elbow in a pipe run (degree 2 in the GT graph, per verified
  schema — see `docs/phase4_design/crossing_convention.md`). In the SKELETON we detect
  these as degree-2 nodes with a bend angle > 20 deg. Degree-3+ skeleton nodes (T-junction
  branches) correspond to `general` symbol nodes in the GT (not connector topology nodes),
  but in Phase 4 output they are emitted as connector topology nodes and contracted out.
  The distinction does not affect edge P/R evaluation because both `general` and detected
  `connector` nodes are contracted before symbol-to-symbol adjacency is computed.

- **Crossing** = two pipes pass over each other without a physical join. OPEN100 uses the
  bridge-gap convention (verified): the "underpass" pipe has a gap of 31–60px where the
  overpass pipe crosses. In the skeleton: two facing endpoint stubs (nearly collinear,
  pointing toward each other) separated by a white-space gap in the binary image, with a
  perpendicular pipe running continuously through the gap.

`CLAUDE.md` already specifies "classic CV -- erase detected symbols+text, skeletonize,
extract segments (LSD/Hough)" for line extraction. Connector/crossing classification
by skeleton analysis is the direct extension of that approach.

### The connector/crossing algorithm

At each candidate intersection (skeleton node degree >= 3, or cluster of branch
endpoints within radius R = 10px):

1. Compute all branch directions: angle from intersection center to first 5px of each branch.
2. Pair branches into collinear sets: two branches are "collinear" if their angular
   difference is within +/-30 deg of 180 deg (pointing in nearly opposite directions).
3. For each collinear pair (A_dir, B_dir):
   - Sample the binary skeleton along the line segment connecting the last pixel of
     branch A to the first pixel of branch B.
   - If there is a break of >= G_min consecutive off-pixels (gap threshold, 20–65px range):
     this pair is a **crossing** (bridge-gap convention -- one pipe passes over).
   - If no gap: this pair is a **connector** (lines actually join here).
4. Classify the intersection:

| Pattern | Classification |
|---|---|
| Degree 3, one collinear pair | **Connector** (one pipe runs through this bend, one branches off) |
| Degree 4, two collinear pairs, both gapless | **4-way connector** (two pipes genuinely join) |
| Degree 2 facing-pair with gap in [20,65]px | **Crossing** (endpoint stubs of the underpass pipe on either side of the bridge gap) |
| Degree 2, bend angle > 20 deg | Elbow **connector** |

**Gap size note (verified against OPEN100):** The bridge gap in OPEN100 is 31–60px in
original full-sheet pixel coordinates, confirmed from the crossing-node pair separation in
`docs/phase4_design/crossing_convention.md`. Use G_min=20px, G_max=65px to accommodate
both orientations (31–32px vertical crossings; 55–60px horizontal crossings) with margin.

Tunable parameters -- all exposed in `configs/phase4.yaml`:

| Param | Default | Meaning |
|---|---|---|
| `junction_radius_R` | 10 px | Endpoint cluster radius for candidate detection |
| `collinear_angle_tol` | 30 deg | Tolerance for "opposite direction" branch pairing |
| `min_crossing_gap` | 20 px | Minimum gap width to classify a facing endpoint pair as crossing |
| `max_crossing_gap` | 65 px | Maximum gap width (above this: the pipe is broken, not a crossing) |
| `elbow_angle_min` | 20 deg | Minimum bend to classify degree-2 node as elbow connector |

---

## Decision 2: Node Set

### Full node set

| Node type | Source | Label | Graph role |
|---|---|---|---|
| `valve` | YOLO detection, scored supercategory | cls_name (e.g., `gate_valve`) | Symbol node — **persistent** |
| `instrument` | YOLO detection, scored supercategory | cls_name (e.g., `pressure_indicator`) | Symbol node — **persistent** |
| `unknown_fitting` | YOLO detection, unscored class idx | cls_name from model | Symbol node — **persistent** (see §2.1) |
| `off_page` | inlet/outlet predictions OR line endpoints at sheet border | `off_page` | Dangling terminal symbol node — **persistent** |
| `junction` | Skeleton degree ≥ 3 node with no matching detected symbol within bind radius | `junction` | Persistent topology node — analogue of GT `general`; **NOT contracted** |
| `connector` | Skeleton degree-2 bend with angle > 20 deg | `connector` | Transient topology node — **contracted out** |
| `crossing` | Skeleton gap-test: facing degree-1 stubs, gap 20–65px, perpendicular pipe present | `crossing` | Transient topology node — crossing-to-crossing edge deleted first, then **contracted out** |
| `flow_arrow` | YOLO detection | `flow_arrow` | **Transit node** — erased + inserted as graph node at centroid; both skeleton stubs bind to it (same pattern as `unknown_fitting`). Undetected arrows leave the pipe as a continuous edge. |

**GT correspondence:**

| GT node label | Role in GT graph | Phase 4 analogue |
|---|---|---|
| valve, instrumentation, arrow | Scored symbol — persistent | valve, instrument, **flow_arrow (now transit node)** |
| general | T-junction symbol — **persistent** (ignored for scoring) | `junction` |
| inlet/outlet | Off-page connector — persistent | `off_page` |
| tank, pump | Vessel/pump symbol — persistent (ignored for scoring) | `unknown_fitting` (if detected) |
| connector | Topology bend/waypoint — contracted | `connector` |
| crossing | Topology crossing waypoint — contracted (edge deleted first) | `crossing` |

The critical symmetry: **GT `general` = predicted `junction`**. Both are persistent T-junction
nodes that remain in the final graph after contraction. Neither is contracted away. Edge
evaluation must account for paths of the form `sym_A -- junction -- sym_B`.

### Why unknown_fitting nodes must be included

If an unknown_fitting (reducer, strainer, spectacle blind, etc.) sits on a pipe run
between symbol A and symbol B, and we omit it from the node set while still erasing its
bbox, the skeleton will find two stubs: A -> [gap] -> B with no junction node. The branch
terminates at the unknown_fitting's bbox boundary; without a node bound there, the A-B
connection is lost entirely.

By adding `unknown_fitting` as a proper graph node:
- Its bbox is erased before line extraction (the skeleton doesn't trace through it)
- Branch endpoints terminating at the bbox boundary are bound to it as ports
- The fitting becomes a transit node: A -> unknown_fitting -> B, preserving pipe reachability

After connector contraction: if the fitting has exactly 2 ports, it compresses to a direct
edge (A, B). If it has 3+ ports (a tee), it remains as a node in the contracted graph.

### Flow arrows -- direction encoding

Arrow predictions are NOT connectivity nodes. After the contracted symbol graph is built:

- For each `flow_arrow` prediction: find the two spatially closest adjacent symbol nodes
  that share a direct edge in the contracted graph.
- Arrowhead orientation (which short side of the bbox is the tip) determines flow direction.
- Tag the edge: `flow_direction = "sym_A_id -> sym_B_id"`.
- If no adjacent pair found: leave the edge undirected.

With ~56% arrow recall, ~44% of flow directions will be absent. Build the undirected
graph first; add directed edges as a layer once the arrow training-data fix is applied
(see `docs/phase1_8_final.md §D` for fix specification).

---

## Decision 3: Symbol Erasure + Line Extraction

### Preprocessing (before erasure)

**Step 0a -- ROI filter:** Exclude predictions whose center falls inside any GraphML
`background` bbox (title-block/border regions). Drops the 21 title-block FPs confirmed
in the readiness check. Without GT: use a fixed border margin of 4% of sheet dimensions
plus a large-near-border-rectangle heuristic.

**Step 0b -- Centroid-distance NMS dedupe:** For each class, sort predictions by
confidence DESC. Greedily suppress any lower-confidence prediction whose centroid is
within `0.5 * sqrt(pred_w * pred_h)` of a higher-confidence prediction in the same class.
Collapses the 22 tile-seam duplicate detections seen on sheet 10.

### Erasure

**Dilation margin M** (default 8px, in `configs/phase4.yaml`): expand each non-arrow
symbol bbox by M on all four sides. Captures line endpoints that enter the symbol body
before terminating. Too large: masks adjacent pipe runs. Too small: skeleton traces into
symbol interiors. 8px is conservative; can be tuned per class if needed.

**Symbol fill:** Paint all dilated bboxes with the background color. Determine background
color from the modal pixel value in the ROI border zone (typically white for OPEN100 sheets).

**Text/tag fill:** Paint `tag_rectangle_simple` (cls 29) and `tag_rectangle_multiline`
(cls 31) predictions with background color. Also any other detected text-region classes.

**Short-gap bridging (Design Decision — Fix 1):**

Symbols that are spatially adjacent (valve touching an instrument, instrument at a pipe
nozzle) often have centre-to-centre distances of 20–60px.  With an 8px dilation margin
applied to each symbol, the dilated bboxes of adjacent symbols frequently overlap,
completely consuming the pipe segment between them.  Shrinking the dilation margin is
not the fix: a smaller margin lets skeleton noise from symbol-body pixels survive into
the branch graph.

**Decision:** keep the dilation as-is for erasure; add a pre-erasure path recovery step
after endpoint binding.

Algorithm (implemented in `lines.py:find_short_gap_pairs`):
1. For each pair of graph nodes whose **dilated** bbox gap ≤ `short_gap_max_px` (default 50px):
2. Check the **pre-erasure Otsu binary** (binarized from the original un-erased image):
   - If the original bboxes overlap (gap ≤ 0): assume connected (symbols are touching).
   - Otherwise: crop the combined region, mask out both symbol footprints, check for any
     remaining dark (pipe) pixels in the inter-symbol gap.
3. If a path is confirmed: add a direct graph edge between the two nodes, bypassing the
   skeleton entirely for this pair.

Config param `short_gap_max_px: 50` — a 50px dilated-gap threshold catches the
overlapping case and pairs up to ~80px centre-to-centre (for typical 20px-wide symbols
with 8px dilation on each side), which matches the instrument-at-pipe-nozzle geometry
in OPEN100 sheets.

### Line extraction

1. Grayscale -> Gaussian blur (sigma=0.8) -> Otsu threshold -> binary (lines=black, bg=white)
2. Morphological closing (disk 2px) to bridge tiny line gaps; opening (disk 1px) to remove noise
3. Skeletonize: `skimage.morphology.skeletonize()` (Zhang-Suen thinning) -> 1-pixel-wide skeleton
4. Branch extraction: `skan` library extracts the skeleton as branch polylines. Each branch
   is a list of (x, y) pixels from one junction to the next or to a free endpoint.
   Discard branches shorter than `min_branch_len` (default 20px).
5. Endpoint binding: a branch endpoint is "bound" to a symbol if it falls within
   `dilated_bbox + 5px proximity zone`. Bound endpoints are absorbed into the symbol node
   rather than creating free junction nodes.

### Line-type classification (solid vs dashed) -- deferred

PID2Graph's `edge_label = "solid"` is reserved but currently all-solid. Gap-run analysis
along each branch polyline could classify dashed (signal/instrument) lines. Defer to a
later sub-phase; build solid-only graph first.

---

## Decision 4: Evaluation

### GT graph preparation

From the GraphML (`networkx.read_graphml()`), verified schema (see
`docs/phase4_design/crossing_convention.md`):

- **Connector nodes** (all degree 2): bends and waypoints on pipe runs. NOT T-junctions.
- **Crossing nodes** (degrees 1–3): waypoints on the underpass pipe. Pairs of crossing
  nodes that are adjacent to each other (crossing-to-crossing edges) mark crossing events.
  The crossing-to-crossing edge IS the explicit non-connectivity signal.
- **General nodes** (degree varies): T-junction and branching-point annotations. These are
  IGNORED_LABELS (not scored), but are **symbol nodes, not topology nodes** — they
  **PERSIST** after contraction. They are the GT analogue of our predicted `junction` nodes.

**Contraction algorithm (2 steps — topology nodes only):**

1. **Delete crossing-to-crossing edges.** Find all edges (u, v) where both u and v have
   label `crossing`. Remove them. This severs the non-connection signal and leaves each
   crossing node connected only to its pipe route (2 or fewer connectors).

2. **Contract all topology nodes** (connector + crossing) **ONLY**: for each such node C
   with remaining neighbors {N1..Nk}, add edges {(Ni, Nj) for all i != j}, then remove C.
   Standard networkx contraction; no geometric inference needed.
   **Do NOT contract symbol nodes.** Valve, instrumentation, arrow, general, inlet/outlet,
   tank, and pump all persist as vertices in the contracted graph.

**GT symbol graph** = contracted graph with vertices = {valve, instrumentation, arrow,
general, inlet/outlet, tank, pump} (all symbol-tier nodes, including ignored-label ones).
Edges = all pipe connections after topology-node contraction.

**Why general nodes persist and why this matters for evaluation:** A T-junction in the GT
creates edges *into* the junction (e.g., `valve_A -- general_T` and `general_T -- valve_B`),
NOT a direct `valve_A -- valve_B` edge. A predicted direct edge (A, B) that skips the
T-junction is therefore a **False Positive** — the GT graph has A--T and T--B, not A--B.
This is the primary reason the predicted graph must also emit persistent `junction` nodes
at unmatched degree≥3 skeleton positions (Decision 2).

### Crossing-separated pairs

From the ORIGINAL graph (before edge deletion), identify pairs of symbol nodes (sym_i,
sym_j) such that:
- The only path from sym_i to sym_j in the original graph passes through a
  crossing-to-crossing edge.
- Equivalently: after deleting all crossing-to-crossing edges, sym_i and sym_j are in
  different connected components (or still connected only by unrelated paths — check
  that they have NO path in the edge-deleted graph).

These are canonical "must NOT be connected" pairs. Sheet 0 has 31 crossing-to-crossing
edges, yielding a bounded set of crossing-separated symbol pairs for evaluation.

### Prediction-to-GT node alignment

Two-phase matching. Each GT node is consumed at most once across both phases.

**Phase A — Symbol matching (CtrMt@50%)**

For each predicted node of type valve, instrument, unknown_fitting, or off_page:
match to the nearest unmatched GT symbol node of the same supercategory within
radius `0.5 × sqrt(gt_w × gt_h)`. Greedy, highest-confidence prediction first.
Same algorithm as Phase 1 evaluation.

Produces:
- **Matched symbol pairs** (pred_i, gt_i): basis for scored-symbol edge evaluation
- **Unmatched predictions**: node FPs
- **Unmatched GT symbols**: node FNs (symbol misses from Phase 1, expected ~3%)

**Phase B — Junction matching (junction_match_radius_px)**

For each predicted `junction` node (degree≥3 skeleton node with no matched symbol):
match to the nearest unmatched GT `general` node within `junction_match_radius_px`
(config parameter, suggested default 40px). Add `junction_match_radius_px: 40` to
`configs/phase4.yaml`.

Matching rules — all three must hold:
1. **Type-constrained:** a predicted junction may only match a GT `general` node —
   never a valve, instrument, or other symbol type.
2. **One-to-one, greedy nearest-distance:** iterate predicted junctions in order of
   increasing distance to their nearest GT general candidate; match greedily. Each GT
   `general` node is consumed at most once; each predicted junction is used at most once.
3. **Within radius:** reject any candidate whose Euclidean centroid-to-centroid distance
   exceeds `junction_match_radius_px`.

Produces:
- **Matched junction pairs** (pred_j, gt_general_j): used for junction-mediated edge evaluation
- **Spurious junctions**: predicted junctions with no GT general within radius (skeleton
  noise, or real unannotated T-junctions the annotator missed)
- **Unmatched GT general nodes**: GT T-junctions the skeleton failed to produce a junction at

**Why a separate, looser radius:**
GT `general` nodes are 8×8px point annotations. CtrMt@50% of an 8px bbox gives a 4px
radius — far too tight for skeleton branch-points. Skeleton junction localization has
±15–30px slop (depending on branch density and skeletonization scale); annotator
placement has ±10–20px additional slop. A 40px radius absorbs both sources while
remaining small enough to avoid cross-junction confusion: typical nearest-general
separation on OPEN100 sheets is ≥60–80px.

### Edge metrics (on matched node pairs only)

Computing only on matched pairs avoids penalizing edge misses caused by upstream node misses.

| Quantity | Formula |
|---|---|
| Edge TP | predicted edge (u, v) where (gt_u, gt_v) is in GT contracted graph |
| Edge FP | predicted edge (u, v) where (gt_u, gt_v) is NOT a GT contracted edge |
| Edge FN | GT contracted edge (gt_u, gt_v) with no predicted counterpart |
| **Edge precision** | TP / (TP + FP) |
| **Edge recall** | TP / (TP + FN) |
| **Edge F1** | 2PR/(P+R) |

Here (u, v) ranges over edges between **any two matched nodes** — matched symbol pairs
AND matched junction pairs both contribute. A junction-mediated edge `sym_A -- junction_J
-- sym_B` produces two scored edges: (sym_A, junction_J) and (junction_J, sym_B).

**Report Edge F1 in two variants:**

- **F1-full:** Alignment graph includes all matched nodes (Phase A + Phase B).
  Edges between any two matched nodes are evaluated. Requires correct junction placement.
- **F1-sym-only:** Restrict to edges between two Phase-A matched symbol nodes only.
  Junction nodes are treated as transparent relay nodes — a path `sym_A -- (junction) --
  sym_B` is folded into a single edge (sym_A, sym_B) for this metric. Conservative
  lower bound; robust to junction localization failures.

**Diagnostic signal:** If F1-full >> F1-sym-only, the skeleton is connecting symbols
correctly but failing to place T-junction nodes accurately. If both are close, junction
detection is working. Gate criterion (Edge F1 ≥ 0.70) applies to F1-full; F1-sym-only
is reported for diagnosis but does not gate the phase.

### Crossing non-edge rate (the headline metric)

```
Cross-connection error = |{crossing-separated GT pairs that appear in predicted edges}|
                         / |all crossing-separated GT pairs|
```

A naive algorithm that treats all skeleton intersections as connectors scores ~100% error
(all crossing-separated pairs get incorrectly connected). The Phase 4 gate is <= 20%.

This is the metric that makes the project: edge F1 measures whether connected symbols are
found; crossing non-edge rate measures whether the algorithm correctly refuses to connect
symbols that merely share a pipe crossing.

### Per-sheet report format

| Metric | Sheet 0 | Sheet 3 | Sheet 10 |
|---|---|---|---|
| Symbol node recall (CtrMt@50%) | from Phase 1 | | |
| Edge F1 — full (junction-mediated) | | | |
| Edge F1 — sym-only (junction-transparent) | | | |
| Edge precision | | | |
| Edge recall | | | |
| Crossing non-edge rate | | | |
| Crossing nodes detected (of GT count) | /71 | /? | /? |
| GT `general` matched / total | /20 | /? | /? |
| Predicted junctions — spurious (no GT general in radius) | | | |
| Predicted junctions — unmatched GT general | | | |

---

## Decision 5: Pipeline Order

```
STEP 0  LOAD + PREPROCESS
  0a. Load full-sheet PNG
  0b. Load saved SAHI inference JSON (Phase 1 output for the sheet)
  0c. ROI filter: drop predictions whose center is in any background bbox
  0d. Centroid-distance NMS dedupe (per class, threshold 0.5*sqrt(w*h))

STEP 1  SYMBOL NODE SET
  1a. Build node list: {cls_id, cls_name, supercategory, bbox, conf}
  1b. Tag each node: scored / unknown_fitting / off_page / flow_arrow
  1c. flow_arrow nodes: set aside for direction-tagging at Step 7

STEP 2  ERASURE  [src/pidetect/graph/erase.py]
  2a. Dilate all non-arrow symbol bboxes by M=8px
  2b. Fill dilated bboxes with background color
  2c. Fill tag_rectangle class bboxes (cls 29, 31)
  -> Output: symbol-erased image (grayscale)

STEP 3  LINE EXTRACTION  [src/pidetect/graph/lines.py]
  3a. Gaussian blur (sigma=0.8) -> Otsu binarize -> morphological cleanup
  3b. Skeletonize (skimage.morphology.skeletonize)
  3c. Extract branches as polylines via skan
  3d. Discard branches shorter than min_branch_len=20px

STEP 4  JUNCTION ANALYSIS  [src/pidetect/graph/junction.py]
  4a. At each degree>=3 skeleton node (or endpoint cluster within R=10px):
      - Compute branch direction angles
      - Pair collinear branches (+/-30 deg tolerance)
      - Gap-test each collinear pair (G=4px default threshold)
  4b. Emit: CONNECTOR or CROSSING node with pass-through pair metadata

STEP 5  GRAPH CONSTRUCTION  [src/pidetect/graph/build.py]
  5a. Bind branch endpoints to symbol bboxes (dilated bbox + 5px zone)
  5b. Build networkx.Graph:
        nodes = symbols UNION connectors UNION crossings
        edges = skeleton branches (with polyline geometry attribute)
  5c. Tag crossing nodes with {pass_through: [(node_A, node_B), (node_X, node_Y)]}

STEP 6  TOPOLOGY CONTRACTION  [src/pidetect/graph/build.py]
  6a. Delete crossing-to-crossing edges (pairs of CROSSING nodes joined by a
        crossing edge, identified by the gap-test in Step 4)
  6b. Contract all degree-2 CONNECTOR nodes (standard contraction: add edges
        between their two neighbors, remove the node)
  6c. Contract all CROSSING nodes (same standard contraction after edge deletion)
  6d. PRESERVE all JUNCTION nodes (degree>=3 skeleton nodes with no matched
        symbol): these are persistent nodes, NOT contracted out.
        They are the predicted-graph analogue of GT `general` nodes.
  -> Output: NetworkX graph with vertices = {symbol nodes} UNION {junction nodes}

STEP 7  DIRECTION TAGGING  [src/pidetect/graph/build.py]
  7a. For each flow_arrow prediction: find the adjacent matched symbol pair
  7b. If pair shares a direct edge in the contracted graph: tag with flow_direction
  7c. If no matching adjacent pair: skip (record as direction-unresolved)

STEP 8  EXPORT  [src/pidetect/graph/build.py]
  8a. JSON: {nodes: [...], edges: [{u, v, flow_direction?}]}
  8b. GraphML: matching PID2Graph two-tier format for diff/compare

STEP 9  EVALUATION  [src/pidetect/graph/evaluate.py]
  9a. Load GT GraphML; contract topology nodes with crossing semantics
  9b. Match predicted nodes to GT nodes — two phases:
        Phase A: predicted symbol nodes -> GT symbol nodes
                 (CtrMt@50%, type-constrained, greedy max-confidence)
        Phase B: predicted junction nodes -> GT general nodes
                 (nearest-distance within junction_match_radius_px,
                  type-constrained: junction->general only, greedy, one-to-one)
        Report junction-match diagnostics: GT general matched, spurious
        junctions, unmatched GT general nodes.
  9c. Compute edge TP/FP/FN -> precision, recall, F1
  9d. Compute crossing-separated pairs -> crossing non-edge rate
  9e. Write per-sheet report table to docs/phase4_results.md
```

---

## Step 3 Measured Ceiling (2026-07-16)

**Step 3 is complete. Do not retune — step 4 is the next precision lever.**

### Three-sheet results (OPEN100, corridor params 10 px / 0.50 / 3.0)

Params tuned on sheet 3; reported without per-sheet retuning to give honest
generalization numbers. "SG-pairs" = short-gap pairs fired by the corridor check.

| Sheet | Nodes | SG-pairs | TP | FP | FN |    P  |    R  |   F1  |
|------:|------:|---------:|---:|---:|---:|------:|------:|------:|
|     0 |   120 |      134 | 24 | 36 |  7 | 0.400 | 0.774 | 0.527 |
|     3 |    85 |       74 |  9 | 20 |  1 | 0.310 | 0.900 | 0.462 |
|    10 |   208 |      211 | 14 | 15 |  8 | 0.483 | 0.636 | 0.549 |
| **mean** | | | | | | **0.398** | **0.770** | **0.513** |

F1 range 0.462–0.549 (spread 0.087). No systematic degradation on the densest
sheet (sheet 10, 208 nodes). Params tuned on sheet 3 generalize without retuning.

### What the corridor check does

`find_short_gap_pairs` (`src/pidetect/graph/lines.py`) fires a direct A→B edge
when the pre-erasure binary shows a stripe of dark pixels **aligned with A→B**:
continuity ≥ 50 % of 2 px bins inside a 10 px half-width corridor, and the corridor
pixel count is not dominated by perpendicular content (perp annular band ≤ 3×
corridor count). The interpose filter (`_has_no_interposing_node`, t∈[0.15, 0.85])
additionally blocks pairs where a third symbol's centroid lies on the A→B segment.

### Why Step 3 tuning cannot improve further

The surviving FPs fall into two structurally irremovable categories.

**A. GT-reach FPs (42 total, ~76 % of all FPs):** both endpoints ARE connected in
the raw GT graph, but the predicted direct edge is wrong:

- *Crossing-topology (bucket ii):* the pipe visibly runs through the A→B gap but
  passes over a crossing node — the GT correctly omits the direct A→B edge. The
  corridor detects the pipe ink and cannot distinguish "A→crossing→B" from a true
  A→B connection. **Fix: step 4 crossing detection.** Once crossings are located,
  the gap-check can exclude regions containing a crossing node.

- *Through-symbol shortcut (bucket i):* the GT path is A→C→B; the corridor sees the
  full pipe run's ink. The interpose filter catches most of these but misses cases
  where C is too far off-axis to satisfy t∈[0.15, 0.85].

**B. GT-disconnected FPs (20 total, ~24 % of all FPs):** no GT path exists.
A neighbour pipe happens to run parallel to A→B through the gap, leaving an aligned
corridor stripe. No corridor parameter change can distinguish a connecting pipe from
a same-direction-adjacent pipe — that distinction requires junction context (step 4).

Sheet 10 (densest sheet) does NOT produce more corridor FPs: precision on sheet 10
(0.483) is the highest across the three sheets, disconfirming the concern about dense
parallel runs. Sheet 10's higher FN count (8) is from valve→instrument connections
whose pipes are fully erased with no surviving corridor ink, not from misfiring.

### Generalization verdict

**Params generalize. Step 3 is closed.** The step-3 F1 ceiling is ~0.46–0.55
(mean 0.51) from detection + pipe tracing alone. Both remaining FP categories require
step 4 crossing/connector resolution, not more step-3 tuning. Step 4 is the next
implementation target.

---

## Files to create

| File | Purpose |
|---|---|
| `src/pidetect/graph/__init__.py` | Package marker |
| `src/pidetect/graph/erase.py` | Symbol/text erasure; returns masked image |
| `src/pidetect/graph/lines.py` | Binarize -> skeletonize -> branch polylines |
| `src/pidetect/graph/junction.py` | Connector/crossing classification (gap + collinearity) |
| `src/pidetect/graph/build.py` | Graph construction, contraction, direction tagging, export |
| `src/pidetect/graph/evaluate.py` | GT contraction, node alignment, edge P/R/F1, crossing metric |
| `scripts/run_phase4.py` | End-to-end: preds + image -> graph -> eval -> report |
| `configs/phase4.yaml` | All tunable params: M, R, G, min_branch_len, angle_tol |

## Code to reuse

| Source | What is reused |
|---|---|
| `src/pidetect/data/open100.py:parse_graphml()` | GT node loading in `evaluate.py` |
| `src/pidetect/data/open100.py:OUR_VALVE_IDX` etc. | Supercategory sets for node tagging |
| `src/pidetect/detect/evaluate.py:_center_recall()` algorithm | CtrMt@50% node matching logic |
| `src/pidetect/detect/predict.py` output JSON format | Phase 1 predictions loaded in Step 0 |

## New dependencies

| Package | Use |
|---|---|
| `skan` | Skeleton branch extraction from skimage skeleton |
| `scikit-image` | `skeletonize`, morphological ops |
| `networkx` | Already present |
| `opencv-python` | Already present; binarization + morphological ops |

---

## Phase 4 gate

**Go criteria (averaged across sheets 0, 3, 10):**

| Metric | Threshold |
|---|---|
| Edge F1 | >= 0.70 |
| Crossing non-edge rate | <= 0.20 |

Edge F1 >= 0.70 is achievable given 97%+ symbol recall (Phase 1 baseline) and competent
line extraction. Below 0.70 indicates systematic line tracing failure -- masking too
aggressive, skeletonization artifacts, or endpoint binding too loose -- requiring algorithm
fixes before proceeding to Phase 5.

Crossing non-edge rate <= 0.20 means the gap-detection algorithm correctly refuses to
connect at least 80% of crossing-separated symbol pairs. This is realistic for the standard
P&ID bridge-gap drawing convention used in the OPEN100 nuclear reactor sheets. A rate above
0.20 means the algorithm is treating most crossings as junctions, which would make the
connectivity graph unreliable for process analysis.
