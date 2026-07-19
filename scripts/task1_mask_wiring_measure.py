"""Task 1 measurement: mask_all_nodes wiring fix, before vs after, same process.

Builds s3/s4/s6 TWICE per sheet -- once with mask_all_nodes=False (the bug,
reproducing last session's numbers) and once with mask_all_nodes=True,
mask_all_nodes_fallback=False (today's now-wired config) -- so the diff isolates
exactly this one change. Reports:
  1. TP/FP/FN/P/R/F1 before vs after, per sheet.
  2. Which TP pairs were LOST (present before, gone after).
  3. Which of the 8 documented "third-node ink" P3b FP pairs are killed.

Read-only measurement; does not change configs/phase4.yaml or call sites (those
are already wired -- this script independently forces both settings for a clean
side-by-side diff).

Usage:
    PYTHONPATH=src python scripts/task1_mask_wiring_measure.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import networkx as nx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import build_node_set, centroid_nms, erase_symbols, roi_filter
from pidetect.graph.lines import binarize as _binarize_orig, run_step3
from pidetect.graph.junction import run_step4
from pidetect.graph.build import run_step5, run_step6, sym_gid
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, run_step9

from run_phase4_steps03 import _load_bg_bboxes, _build_vessel_nodes, _load_or_infer, _build_vessel_nodes

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]

# The 8 "third-node ink" P3b FPs identified in the prior decompose_p3.py run.
THIRD_NODE_INK_FPS = {
    0: [(20, 22), (29, 34), (88, 108), (95, 111)],
    3: [(50, 73), (52, 67), (52, 76), (53, 75)],
    10: [],
}


def _build_s3_s4(sheet_id, weights, cache_dir, conf, iou, device, cfg, mask_all_nodes, mask_fallback):
    png = RAW_DIR / f"{sheet_id}.png"
    gml = RAW_DIR / f"{sheet_id}.graphml"

    bg_bboxes, img_w, img_h, vessel_entries = _load_bg_bboxes(gml)
    raw_preds = _load_or_infer(sheet_id, png, weights, conf, iou, device, cache_dir)

    preds_after_roi, _ = roi_filter(
        raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=cfg["roi_border_margin_frac"],
    )
    preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=cfg["nms_centroid_frac"])

    nodes = build_node_set(preds_deduped)
    vessel_nodes = _build_vessel_nodes(vessel_entries, start_id=len(nodes))
    all_symbol_nodes = nodes + vessel_nodes

    img_bgr = cv2.imread(str(png))
    _, binary_pre_erase = _binarize_orig(img_bgr, blur_sigma=cfg["blur_sigma"], close_disk_r=0, open_disk_r=0)
    vessel_dil = cfg["erase_dilation_px"] + cfg["vessel_erase_dilation_px"]
    erased_bgr = erase_symbols(img_bgr, nodes, dilation_px=cfg["erase_dilation_px"])
    erased_bgr = erase_symbols(erased_bgr, vessel_nodes, dilation_px=vessel_dil)

    s3 = run_step3(
        erased_bgr, all_symbol_nodes, img_w, img_h,
        blur_sigma=cfg["blur_sigma"],
        close_disk_r=cfg["morph_close_disk"],
        open_disk_r=cfg["morph_open_disk"],
        min_branch_len_px=cfg["min_branch_len_px"],
        endpoint_bind_extra_px=cfg["endpoint_bind_extra_px"],
        off_page_border_px=cfg["off_page_border_px"],
        binary_pre_erase=binary_pre_erase,
        mask_all_nodes=mask_all_nodes,
        mask_all_nodes_fallback=mask_fallback,
        short_gap_max_px=cfg["short_gap_max_px"],
        interpose_tol_px=cfg["short_gap_interpose_tol_px"],
        corridor_half_width_px=cfg["short_gap_corridor_half_width_px"],
        continuity_frac=cfg["short_gap_continuity_frac"],
        perp_reject_ratio=cfg["short_gap_perp_reject_ratio"],
    )
    s4 = run_step4(
        s3.skeleton, s3.branches, s3.endpoints,
        collinear_angle_tol=cfg["collinear_angle_tol"],
        min_crossing_gap=cfg["min_crossing_gap"],
        max_crossing_gap=cfg["max_crossing_gap"],
        elbow_angle_min=cfg["elbow_angle_min"],
        junction_radius_R=cfg["junction_radius_R"],
        binary_raw=s3.binary_raw,
        exclude_rects=bg_bboxes,
    )
    return png, gml, all_symbol_nodes, s3, s4


def _pair_set(graph, pred_to_gt, gt_edge_set, want_tp: bool):
    out = set()
    for u, v in graph.edges():
        gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
        if gt_u is None or gt_v is None:
            continue
        is_tp = frozenset((gt_u, gt_v)) in gt_edge_set
        if is_tp == want_tp:
            ida, idb = int(u.split("_")[1]), int(v.split("_")[1])
            out.add(frozenset((ida, idb)))
    return out


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

    print("=" * 100)
    print("TASK 1 -- mask_all_nodes wiring: BEFORE (False, the bug) vs AFTER (True, fallback=False)")
    print("=" * 100)

    for sheet_id in SHEETS:
        png, gml, nodes_before, s3_before, s4_before = _build_s3_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg,
            mask_all_nodes=False, mask_fallback=True,
        )
        s5_before = run_step5(nodes_before, s3_before.off_page_nodes, s3_before, s4_before)
        s6_before = run_step6(s5_before)
        s9_before = run_step9(s6_before.graph, gml)

        png, gml, nodes_after, s3_after, s4_after = _build_s3_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg,
            mask_all_nodes=cfg["mask_all_nodes"], mask_fallback=cfg["mask_all_nodes_fallback"],
        )
        s5_after = run_step5(nodes_after, s3_after.off_page_nodes, s3_after, s4_after)
        s6_after = run_step6(s5_after)
        s9_after = run_step9(s6_after.graph, gml)

        gt_G, _ = load_gt_contracted(gml)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
        pred_to_gt_before, _ = match_nodes(s6_before.graph, gt_G)
        pred_to_gt_after, _ = match_nodes(s6_after.graph, gt_G)

        tp_before = _pair_set(s6_before.graph, pred_to_gt_before, gt_edge_set, want_tp=True)
        tp_after = _pair_set(s6_after.graph, pred_to_gt_after, gt_edge_set, want_tp=True)
        fp_before = _pair_set(s6_before.graph, pred_to_gt_before, gt_edge_set, want_tp=False)
        fp_after = _pair_set(s6_after.graph, pred_to_gt_after, gt_edge_set, want_tp=False)

        lost_tp = tp_before - tp_after
        gained_tp = tp_after - tp_before
        fixed_fp = fp_before - fp_after
        new_fp = fp_after - fp_before

        print(f"\n--- Sheet {sheet_id} ---")
        print(f"  BEFORE (mask_all_nodes=False): TP={s9_before.edge_tp} FP={s9_before.edge_fp} FN={s9_before.edge_fn} "
              f"P={s9_before.edge_precision:.3f} R={s9_before.edge_recall:.3f} F1={s9_before.edge_f1:.3f}")
        print(f"  AFTER  (mask_all_nodes=True,fallback=False): TP={s9_after.edge_tp} FP={s9_after.edge_fp} FN={s9_after.edge_fn} "
              f"P={s9_after.edge_precision:.3f} R={s9_after.edge_recall:.3f} F1={s9_after.edge_f1:.3f}")
        print(f"  TP lost: {sorted(tuple(sorted(x)) for x in lost_tp)}")
        print(f"  TP gained: {sorted(tuple(sorted(x)) for x in gained_tp)}")
        print(f"  FP fixed (gone after): {len(fixed_fp)} -> {sorted(tuple(sorted(x)) for x in fixed_fp)}")
        print(f"  FP new (appeared after): {len(new_fp)} -> {sorted(tuple(sorted(x)) for x in new_fp)}")

        third_node_pairs = {frozenset(pr) for pr in THIRD_NODE_INK_FPS[sheet_id]}
        killed = third_node_pairs & fixed_fp
        survived = third_node_pairs - fixed_fp
        print(f"  Third-node-ink FPs killed: {len(killed)}/{len(third_node_pairs)} -> {sorted(tuple(sorted(x)) for x in killed)}")
        if survived:
            print(f"  Third-node-ink FPs SURVIVED: {sorted(tuple(sorted(x)) for x in survived)}")


if __name__ == "__main__":
    main()
