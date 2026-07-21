"""Phase 3 pre-design check: predicted vs GT instrument_bubble* box tightness.

Phase 1.7e found predicted valve boxes ~43% smaller than GT (area_ratio median 0.567)
even though center-match (CtrMt@50%) was excellent -- CtrMt only guarantees centers
land close enough, not that the box extent is trustworthy. Before committing to 0px
crop padding for OCR (docs/phase3_design.md), the same measurement is needed for
instrument_bubble* classes (25-28) specifically, since undersized predicted boxes with
0 padding would clip tag characters with no downstream recovery.

Measurement only. For sheets 0, 3, 10:
  1. Reproduce the production preprocessing (roi_filter + centroid_nms) on cached raw
     predictions, filtered to instrument_bubble* classes.
  2. Match against GT instrumentation nodes using the SAME greedy centroid-distance
     matching convention as pidetect.graph.evaluate.match_nodes (threshold =
     0.5*sqrt(gt_w*gt_h), sorted by prediction confidence descending).
  3. For each matched pair, compute area_ratio, width_ratio, height_ratio, and signed
     edge insets (positive = predicted edge sits INSIDE the GT edge -- i.e. undersized
     on that side; negative = predicted edge extends beyond GT on that side).
  4. Report median/IQR/worst-case, plus match-rate bookkeeping (matched / total GT /
     total predicted bubbles) needed for the eval-pairing rule.

Usage:
    PYTHONPATH=src python scripts/measure_bubble_bbox_ratio.py --device cpu
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import networkx as nx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import roi_filter, centroid_nms

from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]
BUBBLE_CLS_IDS = frozenset({25, 26, 27, 28})  # instrument_bubble*, per configs/yolo_baseline.yaml


def _load_gt_instrumentation(gml: Path) -> list[tuple[str, float, float, float, float]]:
    g = nx.read_graphml(str(gml))
    out = []
    for nid, attrs in g.nodes(data=True):
        if attrs.get("label") == "instrumentation":
            out.append((nid, float(attrs["xmin"]), float(attrs["ymin"]),
                        float(attrs["xmax"]), float(attrs["ymax"])))
    return out


def _match(bubble_preds: list[dict], gt_nodes: list[tuple[str, float, float, float, float]]):
    """Greedy CtrMt@50% matching, same convention as pidetect.graph.evaluate.match_nodes."""
    import math
    preds_sorted = sorted(bubble_preds, key=lambda p: -p["conf"])
    used_gt: set[str] = set()
    matches = []  # (pred, gt_tuple)
    for p in preds_sorted:
        px = (p["x1"] + p["x2"]) / 2
        py = (p["y1"] + p["y2"]) / 2
        best_d = float("inf")
        best_gt = None
        for gt in gt_nodes:
            gid, gx1, gy1, gx2, gy2 = gt
            if gid in used_gt:
                continue
            gw, gh = gx2 - gx1, gy2 - gy1
            threshold = 0.5 * math.sqrt(gw * gh)
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            d = math.hypot(px - gcx, py - gcy)
            if d <= threshold and d < best_d:
                best_d = d
                best_gt = gt
        if best_gt is not None:
            used_gt.add(best_gt[0])
            matches.append((p, best_gt))
    return matches


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[idx]
    return {
        "n": n, "median": statistics.median(s), "p25": pct(0.25), "p75": pct(0.75),
        "min": s[0], "max": s[-1],
    }


def _fmt(d: dict) -> str:
    if not d:
        return "n=0"
    return (f"n={d['n']:<4} median={d['median']:>7.3f}  p25={d['p25']:>7.3f}  "
            f"p75={d['p75']:>7.3f}  min={d['min']:>7.3f}  max={d['max']:>7.3f}")


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

    all_matches = []  # (sheet_id, pred, gt)
    per_sheet_counts = {}

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
        per_sheet_counts[sheet_id] = dict(
            n_gt=len(gt_instr), n_pred_bubbles=len(bubble_preds), n_matched=len(matches),
        )
        for pred, gt in matches:
            all_matches.append((sheet_id, pred, gt))

    print("=" * 100)
    print("MATCH-RATE BOOKKEEPING (per sheet)")
    print("=" * 100)
    print(f"{'sheet':<8}{'GT instr.':<12}{'pred bubbles':<14}{'matched':<10}{'unmatched GT':<14}{'unmatched pred':<14}")
    tot_gt = tot_pred = tot_matched = 0
    for sheet_id in SHEETS:
        c = per_sheet_counts[sheet_id]
        unmatched_gt = c["n_gt"] - c["n_matched"]
        unmatched_pred = c["n_pred_bubbles"] - c["n_matched"]
        print(f"{sheet_id:<8}{c['n_gt']:<12}{c['n_pred_bubbles']:<14}{c['n_matched']:<10}{unmatched_gt:<14}{unmatched_pred:<14}")
        tot_gt += c["n_gt"]; tot_pred += c["n_pred_bubbles"]; tot_matched += c["n_matched"]
    print(f"{'total':<8}{tot_gt:<12}{tot_pred:<14}{tot_matched:<10}{tot_gt-tot_matched:<14}{tot_pred-tot_matched:<14}")

    # ------------------------------------------------------------------
    # Per-pair ratio / inset computation
    # ------------------------------------------------------------------
    area_ratios, width_ratios, height_ratios = [], [], []
    left_insets, right_insets, top_insets, bottom_insets = [], [], [], []

    detail_rows = []
    for sheet_id, pred, gt in all_matches:
        px1, py1, px2, py2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        gid, gx1, gy1, gx2, gy2 = gt
        pw, ph = px2 - px1, py2 - py1
        gw, gh = gx2 - gx1, gy2 - gy1
        p_area, g_area = pw * ph, gw * gh

        area_ratio = p_area / g_area if g_area else float("nan")
        width_ratio = pw / gw if gw else float("nan")
        height_ratio = ph / gh if gh else float("nan")
        left_inset = px1 - gx1     # + = pred edge inside (right of) GT left edge
        right_inset = gx2 - px2    # + = pred edge inside (left of) GT right edge
        top_inset = py1 - gy1      # + = pred edge inside (below) GT top edge
        bottom_inset = gy2 - py2   # + = pred edge inside (above) GT bottom edge

        area_ratios.append(area_ratio)
        width_ratios.append(width_ratio)
        height_ratios.append(height_ratio)
        left_insets.append(left_inset)
        right_insets.append(right_inset)
        top_insets.append(top_inset)
        bottom_insets.append(bottom_inset)

        detail_rows.append((sheet_id, gid, area_ratio, width_ratio, height_ratio,
                             left_inset, right_inset, top_inset, bottom_inset))

    print("\n" + "=" * 100)
    print(f"BBOX RATIO / INSET DISTRIBUTIONS (n={len(all_matches)} matched pairs, sheets 0/3/10 pooled)")
    print("=" * 100)
    print(f"{'metric':<16}{'stats'}")
    print(f"{'area_ratio':<16}{_fmt(_stats(area_ratios))}")
    print(f"{'width_ratio':<16}{_fmt(_stats(width_ratios))}")
    print(f"{'height_ratio':<16}{_fmt(_stats(height_ratios))}")
    print(f"{'left_inset':<16}{_fmt(_stats(left_insets))}")
    print(f"{'right_inset':<16}{_fmt(_stats(right_insets))}")
    print(f"{'top_inset':<16}{_fmt(_stats(top_insets))}")
    print(f"{'bottom_inset':<16}{_fmt(_stats(bottom_insets))}")
    print("\n(insets in px; positive = predicted edge sits INSIDE the GT edge -- i.e. that side")
    print(" is undersized relative to GT and would need padding to reach the GT extent.")
    print(" negative = predicted edge extends BEYOND the GT edge on that side.)")

    # Worst-case single-edge inset across all four edges (for padding sizing)
    worst_per_pair = [max(row[5], row[6], row[7], row[8]) for row in detail_rows]
    print(f"\n{'worst single-edge inset per pair':<36}{_fmt(_stats(worst_per_pair))}")

    print("\n" + "=" * 100)
    print("WORST 10 PAIRS BY MAX SINGLE-EDGE INSET (candidates for padding sizing)")
    print("=" * 100)
    detail_rows_sorted = sorted(detail_rows, key=lambda r: -max(r[5], r[6], r[7], r[8]))
    print(f"{'sheet':<7}{'gt_id':<18}{'area_r':<9}{'w_r':<8}{'h_r':<8}{'L':<7}{'R':<7}{'T':<7}{'B':<7}")
    for row in detail_rows_sorted[:10]:
        sheet_id, gid, ar, wr, hr, li, ri, ti, bi = row
        print(f"{sheet_id:<7}{gid:<18}{ar:<9.3f}{wr:<8.3f}{hr:<8.3f}{li:<7.1f}{ri:<7.1f}{ti:<7.1f}{bi:<7.1f}")


if __name__ == "__main__":
    main()
