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
Prioritise sheets that contain the weakest classes identified in Phase 1:
`gate_valve_handwheel`, `gate_valve_actuated_dot`, `gate_valve_actuated_stem`,
`butterfly_valve_partial`, `flow_arrow`.

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
5. Export a dataset YAML to `data/realworld_eval/realworld.yaml` following this template:

```yaml
path: data/realworld_eval
train: images/test   # required by Ultralytics even for test-only sets; reuse test
val:   images/test
test:  images/test

nc: 32
names:
  0: butterfly_valve_open
  1: butterfly_valve_crossed
  2: valve_filled_dot
  3: butterfly_valve_open_v2
  4: butterfly_valve_diamond
  5: check_valve_globe
  6: butterfly_valve_with_actuator
  7: check_valve_nrv
  8: gate_valve_solid
  9: angle_valve
  10: butterfly_valve_partial
  11: globe_valve_solid
  12: gate_valve_open_large
  13: control_valve_diaphragm
  14: globe_valve_circle
  15: ball_valve
  16: gate_valve_handwheel
  17: gate_valve_actuated_stem
  18: gate_valve_actuated_dot
  19: reducer
  20: spectacle_blind_spacer
  21: spring_element
  22: heat_exchanger_strainer
  23: flow_arrow
  24: spectacle_blind_closed
  25: instrument_bubble_large
  26: instrument_bubble_medium
  27: instrument_bubble_small_a
  28: instrument_bubble_small_b
  29: tag_rectangle_simple
  30: tag_square
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
