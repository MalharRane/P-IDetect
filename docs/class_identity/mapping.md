# Class identity — verified mapping (subtask 1.6a)

**Why this file exists:** `configs/yolo_baseline.yaml`'s class names were guessed during
subtask 0.2, not derived from ground truth. The Dataset-P&ID paper (Paliwal et al. 2021,
arXiv:2109.03794) only labels its 32 classes as `Symbol1`..`Symbol32` in Figure 3 — **the
paper itself never assigns semantic names either**, it's just 32 glyphs in a grid with no
caption text. So this isn't "look up the real names" (there are none to look up); it's
"stop pretending we know the real name when we don't."

## Method

1. `scripts/build_class_identity_sheets.py` crops up to 16 ground-truth instances of each
   class index straight from the original (untiled, non-synthetic) labelled images at
   `data/digitize-pid-yolo/DigitizePID_Dataset/`, and tiles them into
   `docs/class_identity/idx_NN.png`. This is what's actually in the training data for
   that index — not a prediction, not a guess.
2. I downloaded arXiv 2109.03794, rendered Figure 3 at 600dpi, and read off all 32 glyphs
   to compare against (see process notes below; the figure image itself is not committed
   here for copyright reasons — cite the paper instead).
3. **Naming rule:** assign a semantic name only when the glyph unambiguously matches a
   standard, widely-recognized P&ID/ISA-5.1 symbol *regardless of which exact valve
   sub-type it might be* (e.g. "this is clearly a check valve" vs. "this is clearly *a*
   valve, but I'd be guessing whether it's gate/globe/ball/plug"). Everything in the
   "bowtie-shape-with-a-small-center-mark" cluster — which is most of indices 0-18 — falls
   into the second bucket: the paper groups these as "structurally very similar" on
   purpose (Fig. 3 caption), and guessing a specific valve type from a ~60px synthetic
   glyph is exactly the mistake that created this subtask. Those stay `Symbol_<N>`
   (paper numbering = index + 1), with the distinguishing visual mark noted instead of a
   fabricated name.
4. For the instrument-bubble cluster (idx 25-30), size alone doesn't distinguish them —
   median pixel size is ~122-125px for *all six*, refuting the original
   `instrument_bubble_large/medium/small_a/small_b` guess outright (see size column).
   What **does** distinguish several of them, empirically, across all 16 sampled
   instances per class: the tag-text prefix is a *constant template* for some classes
   (e.g. every idx-26 instance reads "RO-10/&lt;number&gt;") and *random* for others (idx-25
   mixes GLR/GI/GRI/GRP/GRO/...). That's evidence, not a guess, so it's used in the name.

## Mapping table

| idx | paper | chosen name | n (train+val) | median px (w×h) | visual description |
|---|---|---|---|---|---|
| 0 | Symbol1 | `Symbol_1` | 1656 | 63×62 | Open/hollow bowtie (two triangles tip-to-tip), plain end-caps, no center mark. |
| 1 | Symbol2 | `Symbol_2` | 1810 | 63×62 | Bowtie + small circle-with-cross at the pinch point. |
| 2 | Symbol3 | `Symbol_3` | 1749 | 62×62 | Bowtie + solid filled dot at the pinch point. |
| 3 | Symbol4 | `Symbol_4` | 1768 | 62×62 | Bowtie, **both** triangles solid-filled, no other mark. Visually ~identical to idx 11 at this resolution — flagged, not resolved. |
| 4 | Symbol5 | `Symbol_5` | 1813 | 63×62 | Bowtie + small hollow/open circle at the pinch point. |
| 5 | Symbol6 | `angle_valve` | 1834 | 76×76 | Distinct bent/zigzag body with a stem and filled ball — the angle-valve "pipe turns through the valve" silhouette is unambiguous even though exact trim isn't. (Original config had `angle_valve` at idx 9, which is wrong — see below.) |
| 6 | Symbol7 | `valve_handwheel` | 1668 | 63×63 | Bowtie + vertical stem topped with a small lever/flag actuator. Named for the actuator feature, not the body type. (Original config had `gate_valve_handwheel` at idx 16, the class that failed the Phase 1 recall gate — that index is actually something else entirely, see below.) |
| 7 | Symbol8 | `check_valve` | 1711 | 63×63 | Single triangle + end-cap (not a bowtie at all) — the universal, unambiguous ISA check-valve symbol. |
| 8 | Symbol9 | `Symbol_9` | 1807 | 71×73 | Bowtie with curved/bulging sides (rounded lens shape) instead of straight triangle edges. |
| 9 | Symbol10 | `Symbol_10` | 1701 | 85×85 | Bowtie + small downward arrow/flag marker on top. |
| 10 | Symbol11 | `Symbol_11` | 1756 | 63×62 | Bowtie, asymmetric fill — one triangle hollow, the other solid. |
| 11 | Symbol12 | `Symbol_12` | 1840 | 63×62 | Bowtie, both triangles solid-filled. Visually ~identical to idx 3. |
| 12 | Symbol13 | `Symbol_13` | 1780 | 63×62 | Bowtie, both triangles solid-filled **+** small dot at center. |
| 13 | Symbol14 | `control_valve_diaphragm` | 1730 | 109×109 | Open bowtie + stem + dome/mushroom pneumatic-actuator cap — the textbook ISA control-valve-with-diaphragm-actuator symbol. |
| 14 | Symbol15 | `Symbol_15` | 1724 | 58×58 | Small plain (hollow) circle inline on the pipe, no text. Much smaller than the instrument bubbles below (58px vs ~123px). |
| 15 | Symbol16 | `Symbol_16` | 1820 | 58×58 | Same as idx 14 but filled solid. |
| 16 | Symbol17 | `Symbol_17` | 1737 | 51×79 | Double parallel bar (spectacle-blind-like) with one small hollow circle at one end. **This index — not "handwheel" — is the class flagged as the weakest in `docs/phase1_analysis.md` (old name `gate_valve_handwheel`, recall 0.865); the real handwheel/actuator glyph is idx 6. Re-examine that error analysis with the corrected identity.** |
| 17 | Symbol18 | `Symbol_18` | 1739 | 51×102 | Double parallel bar with a hollow circle at one end and a filled circle at the other. |
| 18 | Symbol19 | `Symbol_19` | 1700 | 49×102 | Double parallel bar with two adjacent circles (one filled, one hollow) on the same side. |
| 19 | Symbol20 | `reducer` | 1715 | 69×69 | Open trapezoid/cone — the standard pipe-reducer symbol. |
| 20 | Symbol21 | `Symbol_21` | **5625** | 33×42 | Plain double parallel bar, no circles. By far the most frequent class (~3x typical) — worth a second look in Phase 2; the frequency is more consistent with a ubiquitous fitting (flange/union) than a specialty spectacle blind. |
| 21 | Symbol22 | `strainer` | 1686 | 62×61 | Zigzag/mesh line inside a small rectangle — classic Y-/basket-strainer symbol. |
| 22 | Symbol23 | `heat_exchanger` | 1687 | 97×82 | Elongated capsule with dashed internal lines and rounded end-caps — shell-and-tube symbol. |
| 23 | Symbol24 | `flow_arrow` | 1657 | 54×53 | Solid filled triangle arrowhead alone on the line — universal flow-direction indicator. |
| 24 | Symbol25 | `Symbol_25` | 1806 | 78×78 | Circle with a diagonal cross (⊗) inline, no text. Paper's last "complex" class; could be a top-view valve, plug, or restriction-orifice marker — not confident enough to commit. |
| 25 | Symbol26 | `instrument_bubble` | 1666 | 124×123 | Circle, two lines of text. Sampled tag prefixes **vary** (GLR, GI, GRI, GRP, GRO, ...) — the generic/catch-all instrument-tag bubble covering many ISA letter-codes. |
| 26 | Symbol27 | `instrument_bubble_RO10` | 1758 | 123×123 | Same shape; all 16 sampled instances read a **constant** "RO-10" prefix (only the loop number varies) — a fixed-template tag, not a random one. |
| 27 | Symbol28 | `instrument_bubble_SDL` | 1766 | 122×122 | Same shape; constant "SDL" prefix across all sampled instances. |
| 28 | Symbol29 | `instrument_bubble_DDL` | 1743 | 123×124 | Same shape; constant "DDL" prefix across all sampled instances. |
| 29 | Symbol30 | `tag_rectangle_simple` | 1620 | 107×106 | Plain rectangle, one line of text ("STA" in every sampled instance). **Verified correct — see the dedicated note below; this is the index the corrective prompt's premise was about.** |
| 30 | Symbol31 | `Symbol_31` | 1723 | 125×124 | Circle with a distinctive double-stroke/hatched border (unlike the plain-bordered bubbles above); 2-line text alternates between "ZLC" and "ZLO" prefixes (not perfectly fixed, so the idx26-28 naming trick doesn't cleanly apply). Border likely encodes a different ISA mounting/instrument-type convention, but I'm not confident enough of the exact meaning to assert it. |
| 31 | Symbol32 | `tag_rectangle_multiline` | 1703 | 125×125 | Rectangle, three stacked lines of text in every sampled instance. |

## The bowtie-valve / `tag_rectangle_simple` question

The prompt's premise was: *"Bowtie valves showing as `tag_rectangle_simple` @1.00 in the
overlay is an index→name mismatch."* I traced this and it does **not** hold up:

- Ground truth: I cropped 10 real index-29 instances directly from
  `data/digitize-pid-yolo/.../labels/train` — every one is an "STA" rectangle (see table
  row above). Index 29 = paper `Symbol30` per the HF dataset's stated indexing
  (`0 is 1`), and that's exactly what's there.
- The **main** 50-epoch model (`runs/detect/train/predict/predictions.json`) predicts
  `tag_rectangle_simple` (cls 29) on real "STA" rectangles in `images/train/0.jpg`,
  correctly.
- The bowtie-valve-mislabeled-as-`tag_rectangle_simple`-at-confidence-~1.00 detections
  trace to **`runs/detect/runs/detect/smoke/predict/predictions.json`**, from the
  "smoke" run — a 1-epoch, CPU sanity-check training run
  (`runs/detect/runs/detect/smoke/args.yaml`: `epochs: 1, device: cpu`), not the real
  model. A model trained for one epoch on CPU outputting confident-but-wrong predictions
  is expected and unremarkable — it is not evidence of an index/name bug at idx 29.

So: **idx 29 is correct as named.** The actual, real problem this subtask fixes is
elsewhere — the gate-valve/actuator family (old names at idx 16-18) and most of the
bowtie cluster (idx 0-13, 16-18) had names invented with no basis, and at least one
(`angle_valve`) was simply placed at the wrong index entirely (true idx 5, old config
had it at idx 9).

## Re-run

```
python scripts/build_class_identity_sheets.py
```
Regenerates all 32 contact sheets and reprints the per-class n / median-size table.
