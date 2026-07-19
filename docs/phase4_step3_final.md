# Phase 4 Step 3 — Frozen Evaluation Record

**Date frozen:** 2026-07-17  
**Config:** `mask_all_nodes: true`, `mask_all_nodes_fallback: false`, `min_branch_len_px: 15`  
**Sheets evaluated:** 0, 3, 10 (OPEN100)

---

## Summary

Step 3 with the C1 locked config (all-node bbox masking, no fallback) and `min_branch_len_px=15` achieves a mean edge F1 of **0.499** (P=0.397 R=0.683) across the three evaluation sheets, with a mean node-match recall of 98.8% and a mean crossing cross-connect rate of 16.7% (all violations, since step-4 junction detection is not yet implemented). Of the 64 total FPs across sheets, 39 (60%) are **S4 topology/crossing FPs** — edges that pass through a GT crossing or connector path that step-4 contraction would eliminate — 19 (29%) are **P3 dashed-signal-line FPs** — short-gap edges firing on a signal line with no GT structural connection, resolvable only with line-type classification from Phase 3 — and 6 (9%) are **OTHER**. The dominant remaining precision lever is therefore **step-4 crossing/connector contraction (S4 is the dominant FP bucket)**: once step-4 detects crossings and contracts them, the S4 FPs are eliminated without touching recall. The 22 FNs decompose as: 18 fully-erased-short-pipe (bbox-touching symbols, correctly handled by short-gap geometry rule), 2 dashed-line / C1-masked (Phase-3 lever), 2 long-gap / skeleton-not-reached (step-4 junction tracing or additional skeleton tuning needed).

---

## Three-Sheet Metrics

| Sheet | Match% | TP | FP | FN | P | R | F1 | Cross% |
|---|---|---|---|---|---|---|---|---|
| 0 | 98.6% | 19 | 34 | 12 | 0.358 | 0.613 | 0.452 | 16.7% |
| 3 | 97.8% | 8 | 13 | 2 | 0.381 | 0.800 | 0.516 | 33.3% |
| 10 | 100.0% | 14 | 17 | 8 | 0.452 | 0.636 | 0.528 | 0.0% |
| **mean** | **98.8%** | | | | **0.397** | **0.683** | **0.499** | **16.7%** |

**Cross% = fraction of GT crossing-separated pairs that appear as a predicted edge
(should be 0%; currently 100% because step-4 is not yet implemented)**

## FP Composition

| Sheet | Total FP | S4 topology | P3 signal-line | OTHER |
|---|---|---|---|---|
| 0 | 34 | 27 (79%) | 6 (17%) | 1 (2%) |
| 3 | 13 | 9 (69%) | 3 (23%) | 1 (7%) |
| 10 | 17 | 3 (17%) | 10 (58%) | 4 (23%) |
| **total** | **64** | **39 (60%)** | **19 (29%)** | **6 (9%)** |

**S4** — predicted edge passes through a GT crossing (should be non-connection);
eliminated by step-4 junction analysis.  
**P3** — short-gap FP on a dashed/signal-line connection (no GT path exists);
requires Phase-3 line-type classification.  
**OTHER** — neighbour-pipe skeleton bleed or unclassified.

## FP Detail (per sheet)

### Sheet 0

| Pred pair | Bucket | SG? | A-class | B-class |
|---|---|---|---|---|
| (sym_111,sym_112) | OTHER | yes | instrument_bubble | instrument_bubble |
| (sym_105,sym_108) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_108,sym_113) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_29,sym_34) | P3 | yes | control_valve_diaphr | control_valve_diaphr |
| (sym_29,sym_84) | P3 | yes | control_valve_diaphr | instrument_bubble |
| (sym_37,sym_100) | P3 | yes | control_valve_diaphr | instrument_bubble |
| (sym_84,sym_85) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_10,sym_31) | S4 | yes | Symbol_4 | control_valve_diaphr |
| (sym_101,sym_103) | S4 | yes | instrument_bubble | instrument_bubble |
| (sym_105,sym_113) | S4 | yes | instrument_bubble | instrument_bubble |
| (sym_12,sym_30) | S4 | yes | Symbol_4 | control_valve_diaphr |
| (sym_12,sym_80) | S4 | yes | Symbol_4 | instrument_bubble |
| (sym_14,sym_33) | S4 | yes | Symbol_4 | control_valve_diaphr |
| (sym_14,sym_88) | S4 | yes | Symbol_4 | instrument_bubble |
| (sym_20,sym_21) | S4 | yes | valve_handwheel | valve_handwheel |
| (sym_20,sym_35) | S4 | yes | valve_handwheel | control_valve_diaphr |
| (sym_20,sym_107) | S4 | yes | valve_handwheel | instrument_bubble |
| (sym_21,sym_23) | S4 | yes | valve_handwheel | valve_handwheel |
| (sym_21,sym_87) | S4 | yes | valve_handwheel | instrument_bubble |
| (sym_21,sym_104) | S4 | yes | valve_handwheel | instrument_bubble |
| (sym_21,sym_35) | S4 | no | valve_handwheel | control_valve_diaphr |
| (sym_22,sym_35) | S4 | yes | valve_handwheel | control_valve_diaphr |
| (sym_22,sym_101) | S4 | yes | valve_handwheel | instrument_bubble |
| (sym_23,sym_28) | S4 | yes | valve_handwheel | control_valve_diaphr |
| (sym_23,sym_107) | S4 | yes | valve_handwheel | instrument_bubble |
| (sym_23,sym_115) | S4 | yes | valve_handwheel | instrument_bubble_RO |
| (sym_30,sym_36) | S4 | no | control_valve_diaphr | control_valve_diaphr |
| (sym_35,sym_98) | S4 | yes | control_valve_diaphr | instrument_bubble |
| (sym_35,sym_113) | S4 | yes | control_valve_diaphr | instrument_bubble |
| (sym_36,sym_105) | S4 | yes | control_valve_diaphr | instrument_bubble |
| (sym_8,sym_27) | S4 | yes | Symbol_4 | control_valve_diaphr |
| (sym_8,sym_82) | S4 | yes | Symbol_4 | instrument_bubble |
| (sym_87,sym_115) | S4 | yes | instrument_bubble | instrument_bubble_RO |
| (sym_92,sym_115) | S4 | yes | instrument_bubble | instrument_bubble_RO |

### Sheet 3

| Pred pair | Bucket | SG? | A-class | B-class |
|---|---|---|---|---|
| (sym_10,sym_66) | OTHER | no | control_valve_diaphr | instrument_bubble |
| (sym_2,sym_59) | P3 | yes | Symbol_4 | instrument_bubble |
| (sym_3,sym_66) | P3 | yes | Symbol_4 | instrument_bubble |
| (sym_9,sym_56) | P3 | yes | control_valve_diaphr | instrument_bubble |
| (sym_0,sym_11) | S4 | no | Symbol_4 | control_valve_diaphr |
| (sym_0,sym_1) | S4 | no | Symbol_4 | Symbol_4 |
| (sym_0,sym_74) | S4 | no | Symbol_4 | instrument_bubble |
| (sym_1,sym_11) | S4 | no | Symbol_4 | control_valve_diaphr |
| (sym_1,sym_74) | S4 | no | Symbol_4 | instrument_bubble |
| (sym_11,sym_74) | S4 | no | control_valve_diaphr | instrument_bubble |
| (sym_12,sym_54) | S4 | yes | control_valve_diaphr | instrument_bubble |
| (sym_5,sym_54) | S4 | yes | Symbol_4 | instrument_bubble |
| (sym_54,sym_74) | S4 | yes | instrument_bubble | instrument_bubble |

### Sheet 10

| Pred pair | Bucket | SG? | A-class | B-class |
|---|---|---|---|---|
| (sym_162,sym_189) | OTHER | no | instrument_bubble | instrument_bubble |
| (sym_33,sym_50) | OTHER | yes | Symbol_12 | control_valve_diaphr |
| (sym_34,sym_45) | OTHER | yes | Symbol_12 | control_valve_diaphr |
| (sym_38,sym_162) | OTHER | no | control_valve_diaphr | instrument_bubble |
| (sym_159,sym_160) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_159,sym_178) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_160,sym_177) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_160,sym_178) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_160,sym_190) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_164,sym_167) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_169,sym_183) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_177,sym_178) | P3 | yes | instrument_bubble | instrument_bubble |
| (sym_5,sym_19) | P3 | yes | Symbol_4 | valve_handwheel |
| (sym_5,sym_162) | P3 | yes | Symbol_4 | instrument_bubble |
| (sym_155,sym_185) | S4 | yes | instrument_bubble | instrument_bubble |
| (sym_159,sym_177) | S4 | yes | instrument_bubble | instrument_bubble |
| (sym_159,sym_190) | S4 | yes | instrument_bubble | instrument_bubble |

## FN Composition

| Sheet | Total FN | ERASED | DASHED | LONG |
|---|---|---|---|---|
| 0 | 12 | 9 | 2 | 1 |
| 3 | 2 | 1 | 0 | 1 |
| 10 | 8 | 8 | 0 | 0 |
| **total** | **22** | **18** | **2** | **2** |

**ERASED** — bbox-touching pair; connection fully inside combined erasure zone;
correctly caught by short-gap geometry rule when it is a TP.  
**DASHED** — within short-gap range (dil_gap ≤ 50px) but C1 masking kills
the corridor ink; likely dashed/signal line. Phase-3 lever.  
**LONG** — dil_gap > 50px; needs skeleton tracing or step-4 junction merging.

## Config Snapshot (locked)

```yaml
# Phase 4 — Connectivity Graph Extraction
# All tunable parameters. Referenced by scripts/run_phase4_steps03.py and later step scripts.

# --- Step 0: ROI filter + centroid-NMS ---
roi_border_margin_frac: 0.04  # fallback border margin when GraphML has no background bbox
nms_centroid_frac: 0.5        # suppress lower-conf pred if centroid within frac*sqrt(w*h) of kept pred

# --- Step 2: Erasure ---
erase_dilation_px: 8          # expand each symbol bbox by M px on all 4 sides before fill

# --- Step 3: Line extraction ---
blur_sigma: 0.8               # Gaussian blur sigma before Otsu binarization
morph_close_disk: 2           # closing disk radius (bridge tiny line gaps in px)
morph_open_disk: 1            # opening disk radius (remove noise pixels)
min_branch_len_px: 15         # discard skeleton branches shorter than this
                              # (lowered from 20: a ~17px surviving stub in a 20px dil-gap — e.g.
                              #  sym_14↔sym_8 on sheet 3 — was filtered before binding; 15px catches it)
endpoint_bind_extra_px: 5     # extra px beyond dilated bbox edge for endpoint binding
off_page_border_px: 30        # unbound endpoint within this many px of sheet edge -> off_page node

# --- Step 4: Junction analysis (not yet implemented — params reserved) ---
junction_radius_R: 10         # endpoint cluster radius for candidate intersection (px)
collinear_angle_tol: 30       # tolerance for "opposite direction" branch pairing (degrees)
min_crossing_gap: 5           # min bridge-gap to classify facing stubs as crossing (px)
max_crossing_gap: 110         # max bridge-gap (above = broken pipe, not crossing) (px)
elbow_angle_min: 20           # min bend angle to classify degree-2 node as elbow connector (degrees)

# --- Step 2b: Vessel body erasure (GT-assisted, evaluation pipeline only) ---
# Vessels (tank, pump -- IGNORE_OPEN100_LABELS) are not detected by YOLO. For the
# OPEN100 evaluation we read their bboxes directly from the GT GraphML and treat them
# as SymbolNodes so their outlines get erased and pipe stubs bind to them.
vessel_erase_dilation_px: 12      # extra bbox expansion (px) applied when erasing GT vessel bodies
                                  # (stacked on top of erase_dilation_px; vessels need more margin)

# --- Short-gap bridging ---
# When two symbol dilated-bboxes are within short_gap_max_px, the connecting pipe may
# have been fully consumed by erasure.  We require a directionally-aligned stripe of
# dark pixels in the pre-erasure binary along the A->B corridor.
# See _has_aligned_corridor_path in src/pidetect/graph/lines.py.
#
# Production floor (C1):
#   mask_all_nodes=True, mask_all_nodes_fallback=False
# Rationale: unmasked fallback re-fires ~30 category-O FPs per sheet (ink from third
# symbol bodies / pipe runs in the A-B corridor). Clean separation from the 1 recoverable
# TP (sym_12↔sym_74, sheet 3) is not achievable with ink alone; provenance census
# confirmed corridor ink touches both A and B for 12 FP pairs as well.
# Recovery path: sym_12↔sym_74 uses a dashed *signal line*, distinguishable from solid
# process pipes once Phase 3 (PaddleOCR / line-type classification) is available.
# Until then this TP is intentionally sacrificed for precision.
mask_all_nodes: true            # clear ALL symbol dilated bboxes from corridor binary
mask_all_nodes_fallback: false  # DO NOT re-run on unmasked binary (see note above)
short_gap_max_px: 50           # max gap (px) between dilated bboxes to consider pair
short_gap_interpose_tol_px: 20 # reject pair if a third node interposes on the A->B segment
short_gap_corridor_half_width_px: 10 # half-width (px) of the along-axis sampling corridor
short_gap_continuity_frac: 0.5       # min fraction of 2px along-axis bins with corridor pixels
short_gap_perp_reject_ratio: 3.0     # reject if perp annular band has > ratio * corridor pixels

# --- Step 7: Direction tagging ---
arrow_max_dist_px: 200        # max distance (px) from arrow centre to sym-to-sym segment for assignment

# --- Step 9: Evaluation ---
junction_match_radius_px: 40  # Phase B: predicted junction -> GT general max centroid distance (px)

# --- Diagnostic render ---
render_scale: 0.35            # downsample factor for diagnostic JPEG (full sheets are ~7000px wide)

```

---
*Step 3 frozen. Next: step-4 junction analysis (crossing detection).*