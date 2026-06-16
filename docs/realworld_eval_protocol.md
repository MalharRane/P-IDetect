# Real-World Evaluation Set — Protocol

This directory holds a small, hand-labelled set of real P&ID sheets that are
**out-of-distribution** relative to the HF training data. Its purpose is to
give an honest estimate of how the model generalises beyond the Digitize-PID
symbol library.

**NDA rule:** Only use sheets that are explicitly public-domain or
openly licensed. Never commit the hackathon diagrams or any client material.

---

## Target size

5–10 full sheets, covering as many of the 32 symbol classes as possible.
Prioritise sheets that contain the weakest classes identified in Phase 1
(`docs/phase1_analysis.md`). **Note (subtask 1.6a):** those weak classes were
named under the old guessed-names scheme. Translate via
`docs/class_identity/mapping.md` before using them to pick sheets -- e.g. the
class that failed the recall gate (old name `gate_valve_handwheel`, idx 16)
is verified to actually be `Symbol_17` (a spectacle-blind-like double-bar
symbol, not a handwheel valve at all).

---

## Candidate sources (public, no NDA)

| Source | URL / Access | Notes |
|---|---|---|
| PID2Graph dataset | Request from paper authors (Sanchez-Lengeling et al.) | First public graph-GT set; a subset of ~20 sheets |
| ISO/IEC sample P&IDs | Published in ISO 10628-2 annexes | Check your institution's access |
| Wikimedia P&ID diagrams | Search commons.wikimedia.org for "P&ID" | Variable quality; check licence |
| Open-source plant designs | e.g. DWSIM, OpenModelica example P&IDs | Programmatically generated; clean lines |

---

## Labelling protocol

1. **Format:** YOLO-format `.txt` files (one box per line: `cls xc yc w h`, normalised 0–1).
2. **Tool:** Use [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com)
   (free tier) with the 32 class names from `configs/yolo_baseline.yaml`.
3. **Tile first, label on tiles:** Run `python -m pidetect.data.tiling` on each sheet at
   640 px / 20% overlap, then label the tiles (same resolution the model sees).
   Place outputs in:
   ```
   data/realworld_eval/images/test/   ← tile images
   data/realworld_eval/labels/test/   ← YOLO label files
   ```
4. **Do not label train/val splits** — this set is test-only.
5. A dataset YAML already exists at `data/realworld_eval/realworld.yaml` with the
   verified class names from `docs/class_identity/mapping.md` (subtask 1.6a) --
   use that file directly rather than retyping the template below. It looks
   like this:

```yaml
path: data/realworld_eval
train: images/test   # required by Ultralytics even for test-only sets; reuse test
val:   images/test
test:  images/test

nc: 32
names:
  0: Symbol_1
  1: Symbol_2
  2: Symbol_3
  3: Symbol_4
  4: Symbol_5
  5: angle_valve
  6: valve_handwheel
  7: check_valve
  8: Symbol_9
  9: Symbol_10
  10: Symbol_11
  11: Symbol_12
  12: Symbol_13
  13: control_valve_diaphragm
  14: Symbol_15
  15: Symbol_16
  16: Symbol_17
  17: Symbol_18
  18: Symbol_19
  19: reducer
  20: Symbol_21
  21: strainer
  22: heat_exchanger
  23: flow_arrow
  24: Symbol_25
  25: instrument_bubble
  26: instrument_bubble_RO10
  27: instrument_bubble_SDL
  28: instrument_bubble_DDL
  29: tag_rectangle_simple
  30: Symbol_31
  31: tag_rectangle_multiline
```

---

## Running evaluation

Once the set is labelled and the YAML is in place:

```bash
PYTHONPATH=src python -m pidetect.detect.evaluate \
    --weights runs/detect/train/weights/best.pt \
    --realworld \
    --split test
```

The `--realworld` flag points evaluate.py at `data/realworld_eval/realworld.yaml`
instead of the default HF test split.

---

## Interpreting results

Compare the per-class AP table from `--realworld` against the HF test-split table
in `docs/phase1_analysis.md`. A drop of more than 0.1 in mAP50 on any class flags
a real generalisation gap and becomes the primary target for the next training iteration.
