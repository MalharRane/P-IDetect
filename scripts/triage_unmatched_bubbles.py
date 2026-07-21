"""Phase 3 Task 1 follow-up: correctly triage every "unmatched" instrument_bubble*
prediction on sheets 0/3/10.

A first-pass triage (this session) classified all 14 unmatched predictions by "is
there a real bubble visible at this location" and found 12 real bubbles + 2 hexagonal
non-instrument false positives -- then wrongly concluded the 12 were bubbles GT had
never annotated. That check was incomplete: it never asked whether that same physical
bubble was ALREADY matched to a GT node via a DIFFERENT prediction. `centroid_nms` in
src/pidetect/graph/erase.py dedupes per class (`by_class: dict[cls_id, ...]`), so two
overlapping predictions of the SAME bubble under two different instrument_bubble*
subtype classes both survive NMS; the higher-confidence one gets matched normally and
the lower-confidence one is left "unmatched" even though its GT node is already scored.

This script does the correct check: for every unmatched prediction, find its nearest
GT instrumentation node within the standard CtrMt@50% threshold REGARDLESS of whether
that GT node is already claimed by another prediction, and report whether it is:
  - a duplicate of an already-matched GT node (bucket 3 -- NOT a new eval-set entry,
    NOT a GT annotation gap; a cross-class NMS defect), or
  - genuinely unassociated with any nearby GT instrumentation node (bucket 1: real
    false positive, or bucket 2: genuine GT miss -- needs visual confirmation either
    way, see docs/phase3_eval/crops/sheet3_falsepos*.png for the 2 confirmed FPs this
    session).

Read-only measurement. No OCR. No code under src/ touched.

Usage:
    PYTHONPATH=src python scripts/triage_unmatched_bubbles.py --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import roi_filter, centroid_nms
from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer
from measure_bubble_bbox_ratio import _load_gt_instrumentation, _match, BUBBLE_CLS_IDS

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

    totals = {"duplicate": 0, "no_nearby_gt": 0}

    for sheet_id in SHEETS:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        bg_bboxes, img_w, img_h, _ = _load_bg_bboxes(gml)
        raw_preds = _load_or_infer(sheet_id, png, weights, args.conf, args.iou, args.device, cache_dir)
        preds_after_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=cfg["roi_border_margin_frac"],
        )
        preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=cfg["nms_centroid_frac"])
        bubble_preds = [pr for pr in preds_deduped if pr["cls_id"] in BUBBLE_CLS_IDS]
        gt_instr = _load_gt_instrumentation(gml)
        matches = _match(bubble_preds, gt_instr)
        matched_ids = {id(p) for p, _ in matches}
        matched_gt_ids = {gt[0] for _, gt in matches}

        print(f"\n--- sheet {sheet_id} ---")
        for pr in bubble_preds:
            if id(pr) in matched_ids:
                continue
            px, py = (pr["x1"] + pr["x2"]) / 2, (pr["y1"] + pr["y2"]) / 2
            best_d, best_gt, best_used = float("inf"), None, None
            for gid, gx1, gy1, gx2, gy2 in gt_instr:
                gw, gh = gx2 - gx1, gy2 - gy1
                gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                d = math.hypot(px - gcx, py - gcy)
                thr = 0.5 * math.sqrt(gw * gh)
                if d <= thr and d < best_d:
                    best_d = d
                    best_gt = gid
                    best_used = gid in matched_gt_ids
            if best_gt and best_used:
                status = f"DUPLICATE of already-matched {best_gt}"
                totals["duplicate"] += 1
            elif best_gt:
                status = f"WOULD-MATCH {best_gt} but it's unclaimed -- matching bug, investigate"
            else:
                status = "NO nearby GT instrumentation node -> false positive or genuine GT miss (visual check required)"
                totals["no_nearby_gt"] += 1
            print(f"  {pr['cls_name']:<24} conf={pr['conf']:.3f} "
                  f"bbox=({pr['x1']},{pr['y1']},{pr['x2']},{pr['y2']})  {status}")

    print(f"\nTOTALS: {totals['duplicate']} cross-class duplicates of already-matched GT nodes, "
          f"{totals['no_nearby_gt']} with no nearby GT node at all (need visual confirmation to "
          f"split false-positive vs genuine-miss).")


if __name__ == "__main__":
    main()
