# Phase 3 Design — Tag/Text Extraction (v1: Instrument-Bubble Tags Only)

**Date:** 2026-07-19
**Status:** Design only. No code written against this doc yet.
**Precondition:** Phase 4 (connectivity) is frozen — see `docs/phase4_final.md`.

---

## 0. Scope decision

**v1 ships instrument-bubble tags ONLY**: the text printed inside a detected
`instrument_bubble*` symbol (classes 25–28: `instrument_bubble`,
`instrument_bubble_RO10`, `instrument_bubble_SDL`, `instrument_bubble_DDL` — see
`configs/yolo_baseline.yaml`).

**Explicitly OUT of scope for v1** (propose as v2 only if v1 succeeds):
- **Line tags** — the pipe-spec labels running along process lines
  (`SIZE-MSS-140076-SPEC-HC-X` etc., visible throughout the Phase 4 crops in
  `docs/phase4_step4_scope/`).
- **Title block** text.
- **Notes / callouts** anywhere else on the sheet.
- The already-detected `tag_rectangle_simple` / `tag_rectangle_multiline` classes
  (idx 29, 31) are erased in Phase 4 (`_TAG_RECT_IDX` in `erase.py`) but produce no
  graph node and are not read this phase. They are the natural v2 entry point for
  line tags (the detector already localizes them; only OCR + binding is missing).

### Why instrument bubbles first, concretely

1. **Association is free.** A bubble's text lives inside that bubble's own detected
   bbox — one crop, one OCR call, one node. There is no separate matching/binding
   algorithm to design or validate. Line tags require solving a new "which line does
   this floating text box belong to" problem — structurally the same *kind* of
   nearest-neighbor/geometry question Phase 4 spent three iterations getting right
   for short-gap connectivity (config-wiring bug → bbox-touch bypass → stub-direction
   discriminator, see `docs/phase4_final.md`). Deferring that risk to v2 instead of
   bundling it into the first OCR pass is the same "ask rather than guess" / gate
   discipline CLAUDE.md asks for.
2. **The format is regular and self-validating.** ISA-5.1 bubbles are (almost) always
   a 2-line tag: function letters over a loop number, sometimes with a divider line
   between them (confirmed by direct visual sampling this session — §2). That
   regularity gives a cheap structural validity check (regex on each line) that line
   tags, title blocks, and notes don't have.
3. **The node population is already known-good.** Phase 1 measured instrument
   CtrMt@50% at 99.4% (`docs/phase1_8_final.md`) — essentially every GT instrument
   bubble already has a matching predicted bbox to crop and OCR. v1 risk is only the
   OCR step itself, not a detection-quality unknown.
4. **The eval set is bounded and already exists.** The 3 sheets already used as the
   fixed OPEN100 dev set for Phases 1 and 4 (0, 3, 10) contain 106 GT instrumentation
   nodes total (§4) — enough for a real accuracy/CER number without inventing new
   infrastructure.

---

## 1. OCR engine: PaddleOCR

Per CLAUDE.md's architecture (`Text/OCR: PaddleOCR for tag detection + recognition`) —
affirmed here with concrete justification rather than re-litigated.

### Comparison

| Engine | Fit for small tight bubble crops | Cost |
|---|---|---|
| **PaddleOCR (PP-OCRv4 mobile)** | Strong. Runs in its **default det=True + rec=True mode** on each bubble crop: PaddleOCR's own text-line detector finds however many line boxes are present (normally 2 — function code, loop number) and hands each to the recognizer. We do NOT use `det=False` (recognition-only) — see the correction below, §3 needs line-level boxes to sort top/bottom, which only the detector produces. What we *do* get for free from YOLO's 99.4% CtrMt is not skipping detection, but skipping any need for scene-level/whole-sheet text detection: PaddleOCR only ever sees a single pre-localized bubble, never the full sheet. Actively maintained, good accuracy on clean machine-printed drafting fonts. | Mobile det+rec models ~10–15 MB combined. `paddleocr`/`paddlepaddle` are already reserved (commented) in `requirements.txt` — uncomment for Phase 3. CPU inference is fine (crops are tiny, ~30–40 per sheet); GPU (Colab/Kaggle T4) makes it negligible (<1s/sheet either way). |
| Tesseract | Weak without heavy tuning. Historically struggles on small (<20px line height) text and drafting fonts without per-crop page-segmentation-mode tuning (`--psm 6/7`); no built-in confidence-scored multi-line layout the way PaddleOCR returns it. | Free, ubiquitous, but the accuracy gap on this input class isn't worth the simplicity. |
| EasyOCR | Decent accuracy, but heavier model, slower on CPU, less convenient recognition-only path. No clear advantage over PaddleOCR here. | Larger download, slower CPU inference. |
| TrOCR (transformer, HF) | Powerful in general but overkill for short machine-printed strings; meaningfully slower without a GPU, more moving parts (HF model download, tokenizer) for no accuracy benefit on this input. | Heaviest option; GPU-recommended. |

**Decision: PaddleOCR, PP-OCRv4 mobile models, det=True + rec=True on each
pre-localized bubble crop** (§3). No reason to relitigate the CLAUDE.md choice. The
concrete payoff of already having 99.4%-accurate bubble bboxes from Phase 1 is that
PaddleOCR's detector only ever has to find ~2 short lines inside a small, single-symbol
crop — never a full sheet — not that detection is skipped outright.

> **Correction (this revision):** the previous draft of this section said
> "recognition-first (`det=False`)," which contradicted §2/§3 (both require `det=True`
> to get separate line boxes for top/bottom sorting). That was an error in the
> original draft, not a real design option under consideration — fixed here so the
> doc is internally consistent. `det=False` is not used anywhere in this design.

---

## 2. Crop strategy

### Source image
Crop from the **original, pre-erasure, full-resolution sheet image** — not the
Otsu-binarized or skeletonized artifacts Phase 4 produces. Those are tuned for line
geometry (`blur_sigma`, morphological close/open) and actively destroy the glyph
detail OCR needs.

### BBox and padding — corrected this revision with predicted-box measurement

The previous draft validated 0px padding against **GT** bboxes only. That was the wrong
population to check: production crops **predicted** bubble boxes, and Phase 1.7e found
predicted *valve* boxes ~43% smaller than GT (area_ratio median 0.567) despite
excellent center-match — CtrMt@50% only guarantees centers land close enough, not that
the box extent is trustworthy. If bubble predictions were similarly undersized, 0
padding would clip characters with nothing downstream able to recover them. This needed
checking before any OCR code, so I measured it directly.

**Method:** for sheets 0/3/10, ran the production preprocessing (`roi_filter` +
`centroid_nms`, same config as the connectivity pipeline) on cached raw predictions,
filtered to `instrument_bubble*` (cls 25–28), and matched against GT `instrumentation`
nodes using the exact same greedy CtrMt@50% convention as
`pidetect.graph.evaluate.match_nodes` (nearest centroid within `0.5·√(gt_w·gt_h)`,
sorted by confidence descending). 106/106 GT instrumentation nodes matched (100% —
14 extra predicted bubbles had no GT counterpart, see §4 for how those are handled).

**Result — bubbles behave OPPOSITE to valves.** Predicted boxes are slightly *larger*
than GT, not smaller:

| Metric | median | p25 | p75 | min | max |
|---|---|---|---|---|---|
| area_ratio (pred/GT) | 1.128 | 1.089 | 1.156 | 0.989 | 2.258 |
| width_ratio | 1.054 | 1.032 | 1.073 | 0.955 | 1.337 |
| height_ratio | 1.068 | 1.040 | 1.087 | 1.010 | 1.861 |
| left_inset (px) | −2.00 | −2.56 | −1.38 | −7.45 | +0.37 |
| right_inset (px) | −0.10 | −0.70 | +0.64 | −6.42 | +4.13 |
| top_inset (px) | −2.10 | −2.72 | −1.70 | −11.19 | −0.51 |
| bottom_inset (px) | −0.44 | −0.91 | +0.02 | −22.13 | +1.00 |
| **worst single-edge inset per pair** | **+0.08** | −0.30 | +0.70 | −6.20 | **+4.13** |

(inset = how far a predicted edge sits inside the GT edge; positive = undersized on
that side, negative = predicted box extends beyond GT on that side — see
`scripts/measure_bubble_bbox_ratio.py`.) All four edges have negative medians: on a
typical bubble, the predicted box **already extends past GT on every side**. Across
all 106 matched pairs and 4 edges (424 values), exactly one edge on one pair
(`instrumentation102`, sheet 10, right_inset = +4.13px) is undersized by more than 2px.

**Decision: 6px padding, plus third-party masking (not 0px, not a larger unmasked
pad).** 6px clears the observed worst case (4.13px) with headroom for a slightly worse
case on an unseen sheet, at negligible cost — the box is already oversized on average,
so a few extra background px on top is free. I rendered all 8 sanity crops below at
6px + 6× upscale and confirmed text is fully contained in every case, including
`instrumentation102`.

Padding does reopen the neighbor-bleed risk the original (GT-based) analysis found:
of the 8 sanity crops rendered below, **4 show a stray character or arc from a
neighboring bubble** bleeding in at 6px (the same physical pattern as Phase 4's
bbox-touch-no-ink P3b residual — instrument bubbles are frequently drawn touching or
near-touching). Rather than shrinking padding back down (which would reintroduce the
undersized-edge clipping risk this section exists to fix), the fix is the same one
Phase 4 already validated for exactly this tradeoff: **mask any OTHER detected bubble's
bbox that falls inside the padded crop region** (white-fill it) before OCR — the
per-pair third-party masking pattern from `docs/phase4_final.md` §3.1
(`_corridor_ink_check`'s `third_party_nodes` parameter in `src/pidetect/graph/lines.py`),
applied here to bubble crops instead of connectivity corridors. I verified this
directly: masking cleanly removes the neighbor bleed in all 4 affected sanity crops
(including one bubble flanked by neighbors on *both* sides) while leaving the current
bubble's own text untouched — see `docs/phase3_eval/crop_check/*_masked.png`.

**Rule: crop the detected bbox + 6px on all sides; white-fill any other detected
symbol's bbox that intersects the padded region before OCR.**

### 8-crop sanity render (predicted bboxes, sheet 10, 6px pad + 6× upscale)

Sheet 10 chosen as the densest of the three (39 GT instrumentation nodes, all matched).
Selection: the worst 5 pairs by single-edge inset + 3 additional pairs from dense
instrument clusters, to stress-test both the padding and the neighbor-masking claims
above. Files in `docs/phase3_eval/crop_check/`.

| GT node | Tag (read off crop) | Worst edge inset | Text fully contained? | Neighbor bleed at 6px? | Masked version clean? |
|---|---|---|---|---|---|
| instrumentation102 | PIT 11154A | +4.13 (worst in set) | Yes | No | n/a |
| instrumentation52 | TE 11155B | +2.53 | Yes | **Yes** (neighbor "A") | Yes |
| instrumentation98 | PSV 1161 | +2.22 | Yes | No | n/a |
| instrumentation54 | PSC 1112 | +1.94 | Yes | **Yes** (neighbor "B") | Yes |
| instrumentation81 | MOV 1114 | +1.85 | Yes | No | n/a |
| instrumentation20 | FCV 1101 | (typical) | Yes | No | n/a |
| instrumentation29 | TE 11052A | (typical) | Yes | **Yes** (neighbor fragment) | Yes |
| instrumentation55 | TE 11153B | (typical) | Yes | **Yes** (neighbors on both sides) | Yes |

8/8 fully contained at 6px padding; 4/8 show neighbor bleed (all from tightly-packed
instrument clusters, consistent with §-above); masking resolves all 4, including the
two-sided case (`instrumentation55`). This directly supports the "6px + masking" rule
over either "0px" (would have clipped `instrumentation102`) or "6px, no masking" (would
ship with visible neighbor-character contamination on roughly half the sampled dense
cases).

### Upscale
Median 42×42px bubble ⇒ each of the two text lines is only ~15–18px tall in the raw
crop — well below the height OCR engines want (generally 25–32px+ for reliable
recognition). **Upscale 6× before OCR** (LANCZOS or cubic interpolation). Verified
visually this session: 6× upscaling of sample bubbles (`TCV 1408`, `PSV 1405`,
`FCV 1101`) produces clean, sharp character strokes with no visible aliasing artifacts.

### Binarization
**Do not hard-threshold (Otsu) before OCR.** Phase 4's binarization is tuned to
separate line ink from background for skeleton tracing, not to preserve glyph shape —
a hard threshold at small scale risks breaking thin strokes (serifs, the vertical bar
in "1", the closure of "0" vs "O"). Feed PaddleOCR the upscaled grayscale/RGB crop
directly; PaddleOCR's own internal normalization handles contrast. A mild contrast
stretch is acceptable if OCR quality on the hand-labeled set (§4) shows it helps; a hard
binarize does not get tried first.

### The 2-line structure
Confirmed by direct visual sampling (crops of `instrumentation14`, `15`, `20`, `52`
across sheets 0 and 10): ISA-5.1 bubbles are function-code-over-loop-number, sometimes
with a visible horizontal divider line inside the circle. **Don't try to geometrically
detect the divider.** Let PaddleOCR's own multi-line text detection (`det=True`) find
however many line boxes are actually present in the crop, then sort the returned lines
by vertical center (top→bottom) — this is robust to the divider being present, faint,
or absent, and to minor bubble-to-bubble variation in line spacing.

---

## 3. Association + 2-line parsing

### Association rule
**Trivial by construction.** The OCR crop *is* the bubble's own detected bbox, so OCR
output binds 1:1 to the `SymbolNode` (`node_type == "instrument"`) it was cropped from.
No spatial matching, no nearest-neighbor search, no ambiguity to resolve — this is
exactly the complexity line-tag binding would require and instrument tags don't.

### Parsing rule
Run PaddleOCR (det=True, rec=True) on the upscaled crop → an ordered list of
`(line_bbox, text, confidence)`.

- **n == 2 lines** (expected case): sort by line-bbox y-center. Top line is the
  function-code candidate, bottom is the loop-number candidate. Validate:
  - function: `^[A-Z]{1,5}$`
  - loop_number: `^\d{2,6}[A-Z]{0,2}$` (trailing `A`/`B`/etc. common for redundant or
    split-range instruments — observed directly: `TE 11155A` / `TE 11155B` pair)
    **OR** the narrow placeholder pattern `^X{2,6}$` — confirmed present in hand-labeled
    GT this session (`AORV XXX`, `FCV XXX`, `LCV XXX`; 3 of 106 bubbles, all in "typical
    detail" style callouts, not a real loop number). When the placeholder pattern is
    what matched, `parse_status = "ok_placeholder"` instead of `"ok"` — same structural
    validity, but visibly distinct downstream since "XXX" is not a number a consumer
    should treat as a real loop identifier.

  If both validate (either loop_number form): `parse_status = "ok"` or
  `"ok_placeholder"` accordingly.
- **n == 1 line** (lines merged, or a genuinely single-line bubble): attempt a regex
  split on the single string: `^([A-Z]{1,5})[\s\-]?(\d{2,6}[A-Z]{0,2}|X{2,6})$`. If it
  matches, split into function/loop_number, `parse_status = "single_line_split"`
  (regardless of which loop_number form matched — the single-line case is already a
  lower-confidence path, not worth sub-dividing further). If not, keep the raw text
  only, `parse_status = "single_line_unsplit"`.
- **n == 0, or validation fails both ways**: `parse_status = "failed"`. Store whatever
  raw text PaddleOCR returned (possibly empty) for debugging. **Never fabricate a
  function or loop_number** — a failed parse must be visibly failed downstream, not
  silently defaulted.

### Stored per instrument node, always
`tag_raw_text` (concatenation of returned lines), `tag_function`, `tag_loop_number`,
`tag_confidence` (mean of per-line PaddleOCR confidences), `tag_parse_status` ∈
`{ok, ok_placeholder, single_line_split, single_line_unsplit, failed}`. Downstream
consumers (export, future UI) key off `tag_parse_status` rather than trusting
`tag_function`/`tag_loop_number` blindly.

---

## 4. Evaluation

### Checked: does OPEN100/PID2Graph GT carry tag text?

**No.** Confirmed directly against the raw GraphML this session:

```
labels used across all OPEN100 GraphML nodes:
  {'general', 'tank', 'arrow', 'instrumentation', 'valve',
   'background', 'connector', 'crossing', 'inlet/outlet'}
all attribute keys used across ANY node, ANY label:
  {'label', 'xmin', 'ymin', 'xmax', 'ymax'}
```

Every node — including `instrumentation` — carries only a bounding box and a label.
There is no text/tag field anywhere in the ground truth. Per project rule, **we do not
ship an unevaluated OCR stage** — a hand-labeled eval set is required before v1 is
considered done.

### Hand-labeled eval set (completed)

Reuse the same 3 fixed OPEN100 dev sheets already used for Phase 1 and Phase 4 (0, 3,
10) — continuity with existing eval infrastructure, and Phase 1 already measured 99.4%
instrument CtrMt on exactly these sheets, so the predicted-bubble population to OCR is
already known-good.

| Sheet | GT instrumentation nodes |
|---|---|
| 0 | 36 |
| 3 | 31 |
| 10 | 39 |
| **total** | **106** |

Format: `docs/phase3_eval/tags_gt.csv`, columns `sheet_id, gt_node_id, function,
loop_number, raw_tag, source, label_status` — hand-labeled by visually reading each of
the 106 bubble crops once. `raw_tag` is mechanically derived (`f"{function}
{loop_number}"`) from the hand-typed `function`/`loop_number` columns wherever both are
present, verified to reproduce the 2 rows typed by hand before the derivation was
trusted for the other 103. `label_status` ∈ `{ok (102), placeholder (3), unreadable
(1)}` — see "GT label quality" below.

### Eval pairing rule: OCR runs on PREDICTED bubbles, GT has 106 GT nodes

These are two different populations and need an explicit reconciliation rule, not an
implicit one.

- **Matching**: use the same greedy CtrMt@50% matching as §2's bbox measurement
  (`pidetect.graph.evaluate.match_nodes` convention) to pair each predicted
  `instrument_bubble*` box to a GT instrumentation node. Measured this session on
  sheets 0/3/10: 106/106 GT nodes matched, 14 extra predicted bubbles unmatched (120
  total predicted bubbles across the three sheets).
- **Matched GT node**: score normally — run OCR on the matched prediction's crop,
  compare against that GT node's hand-labeled tag string (exact-match + CER, §above).
- **Unmatched GT node** (a real instrument bubble the detector missed or mismatched):
  **counts as a failure**, not excluded from the denominator. The eval set size stays
  fixed at 106 (or however many GT nodes exist in whatever sheets are used) regardless
  of detection performance — the accuracy metric is against "all instruments that
  actually exist," not "all instruments we happened to detect." Excluding unmatched GT
  nodes would let a worse detector produce a *better*-looking OCR accuracy number by
  silently dropping its hardest cases; that's the wrong incentive for a metric meant to
  gate shipping.
- **Unmatched predicted bubble MUST BE TRIAGED, not auto-excluded.** An unmatched
  prediction is not automatically "extra" or "noise" — it needs to be individually
  checked against one of three explanations before deciding what to do with it:
  1. A genuine detector false positive (not an instrument at all).
  2. A real instrument bubble that GT never annotated (an annotation gap) — would need
     adding to the eval set as a `pred_only_real` row.
  3. **A duplicate detection of an ALREADY-matched GT bubble** — the same physical
     bubble produced two (or more) overlapping predictions under different
     `instrument_bubble*` subtype classes, and since `centroid_nms` in
     `src/pidetect/graph/erase.py` dedupes **per class** (`by_class: dict[cls_id, ...]`),
     a cross-class duplicate survives NMS entirely. The higher-confidence one gets
     matched to the GT node normally; the lower-confidence one shows up as "unmatched"
     even though the GT node it belongs to is already scored.

  **This session's actual result, corrected after a first-pass error:** triaging all 14
  unmatched predictions on sheets 0/3/10 found **12 of the 14 fall into bucket 3**
  (cross-class duplicate of an already-matched bubble — confirmed by checking each
  unmatched prediction's nearest GT instrumentation node and whether that node was
  already claimed by a different prediction), and only **2 fall into bucket 1** (two
  hexagonal "DET A/SHT 2" and "DET B/SHT 2" sheet/detail cross-reference symbols,
  confirmed to have no GT instrumentation node anywhere nearby). **Zero fall into
  bucket 2** — a first pass at this triage wrongly called the 12 duplicates "real
  bubbles GT missed" by checking only "is there a real bubble at this location" (yes)
  without checking "is this the SAME bubble as an already-matched GT node" (also yes,
  via a different, usually higher-confidence, cross-class prediction). The eval set is
  **not** expanded — it stays at 106, all of which were already correctly matched.
  `docs/phase3_eval/crops/DEDUP_ISSUE_sheet10_instrumentation62.png` documents one
  case with a THIRD overlapping subtype prediction on the same bubble (three classes,
  one physical object).
- **Bucket-1 false positives (2 confirmed)**: excluded from the accuracy/CER
  denominator (no GT tag string to compare against), but reported as a separate line:
  *"known non-instrument detections correctly rejected by parse validation: N/2"* —
  since neither "DET B" nor "DET A" (top line) is expected to pass the
  `^[A-Z]{1,5}$` function regex cleanly (a 3-letter code followed by a stray "B"/"A"
  token) or the loop-number regex on "SHT 2", this is expected evidence that §3's
  parsing validation self-rejects clearly-non-instrument OCR output without any
  special-cased detection logic.
- **On OPEN100 annotation completeness**: this session's corrected finding is that
  OPEN100's `instrumentation` labels on sheets 0/3/10 are **not** meaningfully
  incomplete — the 100% GT match rate holds. (The opposite claim in a prior draft of
  this section was the first-pass triage error above; retracted.) This is worth
  keeping in mind as a general caution, not a specific finding: **any** future
  precision/recall number computed against OPEN100 GT — including Phase 1's quoted
  CtrMt figures — should be double-checked for the SAME cross-class-duplicate confound
  before treating "unmatched prediction" as synonymous with "false positive." It
  happened not to matter here, but it easily could have.
- **Production consequence (feeds Task 2, not fixed by this doc)**: the 12 confirmed
  duplicates are a real defect independent of GT/eval bookkeeping — in production
  there is no GT to filter against, so an unfixed cross-class duplicate becomes **two
  (or three) separate graph nodes for one physical instrument**, each independently
  cropped and OCR'd, producing redundant/conflicting tag records. This must be fixed
  before Task 2 ships (see the dedup requirement added to §5).

### GT label quality: `label_status` column

Hand-labeling all 106 crops surfaced three cases that need an explicit scoring rule,
not silent inclusion/exclusion:

- **`unreadable`** (1 of 106, sheet 0 `instrumentation112`): the crop has no legible
  function or loop number — blank in the hand-labeled CSV, not a transcription gap.
  **Excluded from the accuracy/CER denominator**, reported as a separate count. A
  bubble no human can read is a GT-quality limitation, not an OCR failure; scoring the
  OCR stage against an unreadable target would penalize the model for something no
  system (human or automated) could get right.
- **`placeholder`** (3 of 106: `AORV XXX` sheet 0, `FCV XXX` sheet 3, `LCV XXX` sheet
  10): a legible tag whose loop number is a literal `XXX` placeholder (typical-detail
  callout convention, not a real loop number). **Scored normally** — these are real,
  readable text PaddleOCR should recognize like any other tag; the only difference is
  the loop-number regex needs a second accepted pattern (`^X{2,6}$`, see §3) so a
  correctly-read placeholder still validates instead of being misclassified as a
  parse failure. `parse_status = "ok_placeholder"` keeps the distinction visible.
- **`ok`** (102 of 106): normal tags, scored normally, `parse_status` unconstrained.

### Metrics

- **Exact-match accuracy** (primary gate metric): normalized reconstructed tag
  (`f"{function} {loop_number}"`, uppercased, whitespace-collapsed) vs. the
  hand-labeled GT string.
- **Character Error Rate (CER)** (secondary): Levenshtein edit distance between
  `tag_raw_text` and the GT raw tag string, micro-averaged (sum of edit distances /
  sum of GT character counts) across the eval set. Catches "close but not exact" OCR
  errors (e.g. `1408` read as `1409`) separately from structural parse failures.
- **Report both metrics split by `tag_parse_status`** — an overall number can hide a
  bad failure rate inside a good exact-match-when-parsed number. Report `ok` +
  `ok_placeholder` + `single_line_split` + `single_line_unsplit` + `failed` counts
  alongside accuracy/CER. `unreadable`-labeled GT rows are excluded from all of the
  above (see "GT label quality") and reported as their own count instead.
- **Proposed (not mandated) gate for shipping v1**: exact-match ≥ 90% on
  `parse_status in {"ok", "ok_placeholder"}` rows; ≥ 80% overall among scoreable rows
  (i.e. excluding `unreadable`, including `failed` rows scored as wrong). Below that:
  iterate on crop/upscale parameters against the hand-labeled set before considering v2
  (line tags) at all.

---

## 5. Integration into graph / export

### Pre-OCR requirement: cross-class instrument-bubble dedup (new, discovered this session)

`centroid_nms` in `src/pidetect/graph/erase.py` dedupes **per class**
(`by_class: dict[cls_id, list[dict]]` — see step 0b). This is correct for genuinely
different symbol classes sitting near each other, but `instrument_bubble` /
`instrument_bubble_RO10` / `instrument_bubble_SDL` / `instrument_bubble_DDL` /
`Symbol_31` are mutually exclusive labels for the **same kind of object** — a physical
bubble is exactly one of these, never two — so a per-class NMS pass provides no dedup
at all when the detector's class head disagrees with itself about which subtype a
bubble is. Confirmed empirically this session: **12 cross-class duplicate detections**
across sheets 0/3/10 (§4), including one bubble (`instrumentation62`, sheet 10) with
**three** overlapping subtype predictions simultaneously (see
`docs/phase3_eval/crops/DEDUP_ISSUE_sheet10_instrumentation62.png`). Left unfixed, each
duplicate becomes a separate graph node, separately cropped and OCR'd, producing
redundant/conflicting tag records for one real instrument.

**Requirement for Task 2, before any OCR runs in production**: add a class-agnostic
dedup pass over `bubble_cls_ids` (configs/phase3.yaml) specifically — treat all four
(five, counting `Symbol_31`) bubble subtypes as one group for centroid-NMS purposes,
keeping the highest-confidence detection per physical bubble (not necessarily the one
with the tightest/most-complete box — the `instrumentation62` crop shows the
highest-confidence match is not always the best-framed one for OCR, a secondary
finding worth revisiting if OCR accuracy on such bubbles is poor). **Then** add a
runtime assertion (fires during graph/node-set construction, not just at eval time)
that no two `node_type == "instrument"` nodes have centroids within the standard
centroid-NMS radius of each other — this is a regression guard, not the fix itself; an
assertion alone would just make Task 2 fail loudly and immediately on this exact
dataset without the dedup pass above.

### Node attributes / export

New node attributes, populated after node-set construction (Phase 4 Step 1) and
independent of erasure/skeleton/junction (Steps 2–9) — **OCR only needs the raw image
+ the instrument bubble bboxes**, so it can run at any point relative to the
connectivity pipeline, including in parallel:

- `tag_function`, `tag_loop_number`, `tag_raw_text`, `tag_confidence`, `tag_parse_status`

**JSON export** (`_node_to_dict` in `src/pidetect/graph/export.py`): add an optional
`"tag"` dict, present only when `node_type == "instrument"`:

```json
{"id": "sym_37", "node_type": "instrument", "cls_name": "instrument_bubble", ...,
 "tag": {"function": "TCV", "loop_number": "1408", "raw_text": "TCV 1408",
         "confidence": 0.97, "parse_status": "ok"}}
```

**GraphML export** (`export_graphml`): add the same fields as extra string node
attributes (`tag_function`, `tag_loop_number`, `tag_raw_text`, `tag_confidence`,
`tag_parse_status`) alongside the existing PID2Graph-compatible `label`/`xmin`/`ymin`/
`xmax`/`ymax`. `nx.write_graphml` tolerates arbitrary extra attributes, and PID2Graph's
own GT schema (and `evaluate.py`'s reader) only ever look at the attributes they know
about — adding tag fields cannot break the existing diff/eval machinery.

No edge-schema change — tags are a node-level enrichment only.

---

## Summary

| Question | Answer |
|---|---|
| v1 scope | Instrument-bubble tags only; line tags/title block/notes → v2 if v1 succeeds |
| OCR engine | PaddleOCR PP-OCRv4 mobile, **det=True + rec=True** on each pre-localized bubble crop (corrected this revision — `det=False` doesn't work, §3 needs per-line boxes) |
| Crop | Original image, detected bbox **+ 6px padding** (measured: predicted boxes are ~13% larger than GT on average, not smaller like valves; worst observed single-edge undersize was 4.13px), **+ mask any other detected bubble's bbox inside the padded crop** (4/8 sanity crops showed neighbor-bleed at 6px; masking resolved all 4), 6× upscale, no hard binarization |
| Association | Free — 1 crop = 1 bubble node, no matching algorithm needed |
| Parsing | 2-line sort top/bottom + regex validation; 1-line regex-split fallback; explicit `failed` status, never fabricated |
| Eval | GT has no tag text (confirmed) → 106-bubble hand-labeled set across sheets 0/3/10, **completed** (not expanded — see below); predicted→GT matched via CtrMt@50% (106/106 matched); **unmatched GT scores as failure**; unmatched predictions **triaged, not auto-excluded** — of 14 this session, 12 were cross-class duplicate detections of already-matched bubbles (not new bubbles, not added to eval set) and 2 were confirmed non-instrument false positives (excluded from scoring, reported as a separate "correctly rejected" line); GT itself has 3 `placeholder` (`XXX` loop number, scored normally via a second accepted regex, `parse_status="ok_placeholder"`) and 1 `unreadable` (excluded from the denominator) row; exact-match + CER, split by parse_status |
| Integration | New node attrs, optional `"tag"` block in JSON export, extra string attrs in GraphML export, no edge changes; **pre-OCR dedup pass required** (class-agnostic NMS across `instrument_bubble*` subtypes) + a construction-time assertion, since 12 cross-class duplicate detections were confirmed on sheets 0/3/10 alone |
