"""Phase 4 step-4 §6c validation gate — read-only, no production change.

For sheet 3, identifies every short-gap pair that is a true positive (its
predicted edge matches a real GT edge), then checks whether the proposed
contracted-skeleton veto (build SR without short-gap edges, then Sc = SR
with connectors/crossings contracted) would ever fire on it.

Gate (docs/phase4_step4_scope.md §6c): every sheet-3 short-gap TP must show
sr_conn=False OR sc_conn=False -> True (i.e. NOT (sr_conn and not sc_conn)).
Any pair with sr_conn=True and sc_conn=False is a RISK: the veto would wrongly
suppress a real edge, and must be diagnosed before step 4 is implemented.

Usage:
    PYTHONPATH=src python scripts/validate_step4_tp_gate.py \\
        --weights runs/detect/train_small_objects/weights/best.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import networkx as nx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import build_node_set, centroid_nms, erase_symbols, roi_filter
from pidetect.graph.lines import binarize as _binarize_orig, run_step3, assert_run_step3_wired
from pidetect.graph.junction import run_step4
from pidetect.graph.build import (
    build_graph, run_step5, run_step6,
    _contract_connectors, _contract_crossings, sym_gid,
)
from pidetect.graph.evaluate import load_gt_contracted, match_nodes

from run_phase4_steps03 import _load_bg_bboxes, _build_vessel_nodes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEET_ID = 3


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

    png = RAW_DIR / f"{SHEET_ID}.png"
    gml = RAW_DIR / f"{SHEET_ID}.graphml"
    cache_dir = REPO / args.cache_dir
    weights = REPO / args.weights

    bg_bboxes, img_w, img_h, vessel_entries = _load_bg_bboxes(gml)
    raw_preds = _load_or_infer(SHEET_ID, png, weights, args.conf, args.iou, args.device, cache_dir)

    preds_after_roi, _ = roi_filter(
        raw_preds, bg_bboxes, img_w, img_h,
        border_margin_frac=cfg["roi_border_margin_frac"],
    )
    preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=cfg["nms_centroid_frac"])

    nodes = build_node_set(preds_deduped)
    vessel_nodes = _build_vessel_nodes(vessel_entries, start_id=len(nodes))
    all_symbol_nodes = nodes + vessel_nodes

    img_bgr = cv2.imread(str(png))
    _, binary_pre_erase = _binarize_orig(
        img_bgr, blur_sigma=cfg["blur_sigma"], close_disk_r=0, open_disk_r=0
    )
    vessel_dil = cfg["erase_dilation_px"] + cfg["vessel_erase_dilation_px"]
    erased_bgr = erase_symbols(img_bgr, nodes, dilation_px=cfg["erase_dilation_px"])
    erased_bgr = erase_symbols(erased_bgr, vessel_nodes, dilation_px=vessel_dil)

    step3_kwargs = dict(
        blur_sigma=cfg["blur_sigma"],
        close_disk_r=cfg["morph_close_disk"],
        open_disk_r=cfg["morph_open_disk"],
        min_branch_len_px=cfg["min_branch_len_px"],
        endpoint_bind_extra_px=cfg["endpoint_bind_extra_px"],
        off_page_border_px=cfg["off_page_border_px"],
        mask_all_nodes=cfg["mask_all_nodes"],
        mask_all_nodes_fallback=cfg["mask_all_nodes_fallback"],
        short_gap_max_px=cfg["short_gap_max_px"],
        interpose_tol_px=cfg["short_gap_interpose_tol_px"],
        corridor_half_width_px=cfg["short_gap_corridor_half_width_px"],
        continuity_frac=cfg["short_gap_continuity_frac"],
        perp_reject_ratio=cfg["short_gap_perp_reject_ratio"],
        stub_angle_tol_deg=cfg["short_gap_stub_angle_tol_deg"],
    )
    assert_run_step3_wired(cfg, step3_kwargs)
    s3 = run_step3(
        erased_bgr, all_symbol_nodes, img_w, img_h,
        binary_pre_erase=binary_pre_erase,
        **step3_kwargs,
    )
    print(f"Sheet {SHEET_ID}: {len(s3.short_gap_pairs)} short-gap pairs total")

    # Step 4 does not exist yet in the pipeline sense of "implemented mechanism", but
    # run_step4 (crossing/connector detection) is already implemented -- only the
    # veto wiring (this gate) is new. Run it to get real SR/Sc inputs.
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
    print(f"Sheet {SHEET_ID}: {s4.n_connectors} connectors, {s4.n_crossings} crossings detected")

    # Contracted pred graph, for matching against GT (mirrors run_step9).
    s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)
    s6 = run_step6(s5)
    gt_G, _crossing_infos = load_gt_contracted(gml)
    pred_to_gt, _gt_to_pred = match_nodes(s6.graph, gt_G)
    gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}

    # Identify short-gap pairs that are TPs: both endpoints matched to GT nodes,
    # and that GT pair is a real GT edge.
    tp_pairs: list[tuple[int, int]] = []
    for id_a, id_b in s3.short_gap_pairs:
        gid_a, gid_b = sym_gid(id_a), sym_gid(id_b)
        gt_a = pred_to_gt.get(gid_a)
        gt_b = pred_to_gt.get(gid_b)
        if gt_a is None or gt_b is None:
            continue
        if frozenset((gt_a, gt_b)) in gt_edge_set:
            tp_pairs.append((id_a, id_b))

    print(f"Sheet {SHEET_ID}: {len(tp_pairs)} short-gap TPs identified\n")

    # Build SR (no short-gap edges) and Sc (SR with connectors + crossings contracted).
    s3_no_sg = dataclasses.replace(s3, short_gap_pairs=[])
    SR, _n_dangling = build_graph(all_symbol_nodes, s3.off_page_nodes, s3_no_sg, s4)
    Sc = SR.copy()
    _contract_connectors(Sc)
    _contract_crossings(Sc)

    print(f"{'pair':<20}{'sr_conn':<10}{'sc_conn':<10}{'verdict'}")
    n_risk = 0
    for id_a, id_b in tp_pairs:
        gid_a, gid_b = sym_gid(id_a), sym_gid(id_b)
        sr_conn = gid_a in SR and gid_b in SR and nx.has_path(SR, gid_a, gid_b)
        sc_conn = gid_a in Sc and gid_b in Sc and nx.has_path(Sc, gid_a, gid_b)
        risk = sr_conn and not sc_conn
        verdict = "SUPPRESS_RISK" if risk else "SAFE"
        if risk:
            n_risk += 1
        print(f"{gid_a}<->{gid_b:<12}{str(sr_conn):<10}{str(sc_conn):<10}{verdict}")

    print()
    if n_risk == 0:
        print(f"GATE PASSED: all {len(tp_pairs)} sheet-{SHEET_ID} short-gap TPs are safe "
              f"(sr_conn=False or sc_conn=True). Proceed to implementation.")
    else:
        print(f"GATE FAILED: {n_risk} of {len(tp_pairs)} TPs show sr_conn=True and "
              f"sc_conn=False. Diagnose before implementing the veto.")


if __name__ == "__main__":
    main()
