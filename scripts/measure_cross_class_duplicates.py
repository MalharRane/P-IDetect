"""Phase 3/4 dedup blast-radius measurement (Task 2 of the cross-class-duplicate fix).

The instrument_bubble* fix showed that centroid_nms (src/pidetect/graph/erase.py)
dedupes PER CLASS, so two overlapping predictions of the SAME physical object under
DIFFERENT classes both survive NMS. Confirmed for instrument bubbles (12 cases across
sheets 0/3/10, one triple). This script checks whether the same defect reaches beyond
bubbles: across ALL classes, for the CURRENT production node set (post per-class
centroid_nms, exactly as run_phase4_steps03.py builds it today), count node pairs of
DIFFERENT classes whose centroids fall within the standard centroid-NMS radius of each
other, per sheet, broken down by class-pair.

A pair this finds is NOT necessarily a duplicate the way bubbles are (two genuinely
different symbols -- e.g. a valve actuator glyph sitting right next to a bubble --
CAN legitimately have close centroids without being the same physical object). This
measurement's job is to surface which class-pairs cluster close together at all, so a
human can judge same-object-duplicate vs legitimate-adjacent-symbols per pair; it does
not itself conclude "these are all duplicates."

Read-only measurement. No fix applied here (see src/pidetect/graph/erase.py for that).

Usage:
    PYTHONPATH=src python scripts/measure_cross_class_duplicates.py --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import roi_filter, centroid_nms, _classify_pred
from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    weights = REPO / args.weights
    cache_dir = REPO / args.cache_dir
    centroid_frac = cfg["nms_centroid_frac"]

    grand_pair_counter: Counter = Counter()
    grand_total = 0

    for sheet_id in SHEETS:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        bg_bboxes, img_w, img_h, _ = _load_bg_bboxes(gml)
        raw_preds = _load_or_infer(sheet_id, png, weights, args.conf, args.iou, args.device, cache_dir)
        preds_after_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=cfg["roi_border_margin_frac"],
        )
        # Current PRODUCTION behaviour: per-class-only NMS (the defect being measured).
        preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=centroid_frac)

        pair_counter: Counter = Counter()
        close_pairs = []
        n = len(preds_deduped)
        for i in range(n):
            pi = preds_deduped[i]
            wi, hi = pi["x2"] - pi["x1"], pi["y2"] - pi["y1"]
            cxi, cyi = (pi["x1"] + pi["x2"]) / 2, (pi["y1"] + pi["y2"]) / 2
            thr_i = centroid_frac * math.sqrt(max(wi * hi, 1.0))
            for j in range(i + 1, n):
                pj = preds_deduped[j]
                if pi["cls_id"] == pj["cls_id"]:
                    continue  # same-class handled by centroid_nms already
                wj, hj = pj["x2"] - pj["x1"], pj["y2"] - pj["y1"]
                cxj, cyj = (pj["x1"] + pj["x2"]) / 2, (pj["y1"] + pj["y2"]) / 2
                thr_j = centroid_frac * math.sqrt(max(wj * hj, 1.0))
                d = math.hypot(cxi - cxj, cyi - cyj)
                # Flag if close relative to EITHER prediction's own box (whichever is
                # more permissive) -- catches near-duplicates regardless of which of
                # the two differently-sized boxes you'd measure the threshold from.
                thr = max(thr_i, thr_j)
                if d < thr:
                    key = tuple(sorted((pi["cls_name"], pj["cls_name"])))
                    pair_counter[key] += 1
                    close_pairs.append((key, pi, pj, d, thr))

        print(f"\n--- sheet {sheet_id}: {len(preds_deduped)} predictions post per-class NMS, "
              f"{sum(pair_counter.values())} cross-class close pairs ---")
        for key, count in pair_counter.most_common():
            print(f"  {key[0]:<26} <-> {key[1]:<26} : {count}")
        for key, pi, pj, d, thr in close_pairs:
            print(f"    [{key}] conf={pi['conf']:.2f}/{pj['conf']:.2f} "
                  f"bbox_i=({pi['x1']},{pi['y1']},{pi['x2']},{pi['y2']}) "
                  f"bbox_j=({pj['x1']},{pj['y1']},{pj['x2']},{pj['y2']}) d={d:.1f} thr={thr:.1f}")

        grand_pair_counter.update(pair_counter)
        grand_total += sum(pair_counter.values())

    print("\n" + "=" * 100)
    print(f"TOTAL cross-class close pairs across sheets 0/3/10: {grand_total}")
    print("=" * 100)
    for key, count in grand_pair_counter.most_common():
        print(f"  {key[0]:<26} <-> {key[1]:<26} : {count}")

    # --- exact category subtotals (code-computed, no hand-tallying) ---
    from pidetect.data.open100 import OUR_INSTRUMENT_IDX, OUR_VALVE_IDX
    cls_name_to_id = {}
    for sheet_id in SHEETS:
        gml = RAW_DIR / f"{sheet_id}.graphml"
        raw_preds = _load_or_infer(sheet_id, RAW_DIR / f"{sheet_id}.png", weights, args.conf, args.iou, args.device, cache_dir)
        for pr in raw_preds:
            cls_name_to_id[pr["cls_name"]] = pr["cls_id"]

    def _node_type(cls_name: str) -> str:
        return _classify_pred(cls_name_to_id[cls_name])

    bubble_total = same_valve_total = other_total = 0
    for (name_a, name_b), count in grand_pair_counter.items():
        ta, tb = _node_type(name_a), _node_type(name_b)
        if ta == "instrument" and tb == "instrument":
            bubble_total += count
        elif ta == "valve" and tb == "valve":
            same_valve_total += count
        else:
            other_total += count

    print("\n" + "-" * 100)
    print("EXACT category subtotals (code-computed):")
    print(f"  instrument_bubble family (both sides node_type='instrument'): {bubble_total}")
    print(f"  valve family (both sides node_type='valve', different classes): {same_valve_total}")
    print(f"  cross-node-type or involving unknown_fitting/arrow/tag_rect: {other_total}")
    print(f"  sum check: {bubble_total + same_valve_total + other_total} (should equal {grand_total})")


if __name__ == "__main__":
    main()
