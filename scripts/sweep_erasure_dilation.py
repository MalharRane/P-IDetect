"""Phase 4 Fix 2 -- erasure-dilation sweep to recover shared-header short-gap FPs.

docs/phase4_final.md S5: the dominant residual (32 of 46 P3 FPs, "P3b shared-header")
is a short-gap pair where A and B each independently tap perpendicularly onto a shared
trunk parallel to the A-B axis -- corridor ink is real but belongs to the trunk, not a
direct A-B line. The stub-direction discriminator (_stub_direction_ok, lines.py) can
reject these, but needs a SURVIVING post-erasure skeleton stub to read a direction
from -- and at erase_dilation_px=8 (current), 0 of the 32 pairs have one (the connecting
tap is fully consumed by erasure before the skeleton is ever traced).

This script sweeps erase_dilation_px down (8/6/4/3/2), re-running the full production
pipeline (steps 0-9, mirroring run_phase4_steps03.py / run_step4_veto_eval.py) on
sheets 0/3/10 for each value, and measures:
  - connectivity TP/FP/FN/P/R/F1 (Step 9, vs the Fix-1 baseline == dilation 8)
  - P3 FP mechanism breakdown (mirrors scripts/decompose_p3.py's bucket taxonomy) --
    "shared-header FPs" == the P3ab(short_gap_ink) bucket specifically (short-gap,
    real corridor ink, not third-node-contaminated, not bbox-touch-zero-ink)
  - "short bound stubs" -- skeleton branches < 2*min_branch_len_px with an endpoint
    bound to a valve/instrument node. This is the measurable proxy for the failure
    mode named in the task ("too little dilation leaves symbol-body pixels that create
    spurious skeleton stubs"): the delta vs the dilation=8 baseline is NEW stub
    population, some of which is real (now-recoverable pipe taps -- the whole point)
    and some of which may be residual glyph-edge ink. Distinguishing the two requires
    a visual spot check, done separately (not a fully-automated split, honestly not
    invented as one).
  - s4.n_connectors / s4.n_crossings totals (coarse false-junction proxy)
  - the exact TP pair-id set per sheet, diffed against the dilation=8 baseline, so any
    TP loss is named, not just counted.

Usage:
    PYTHONPATH=src python scripts/sweep_erasure_dilation.py --device cpu
    PYTHONPATH=src python scripts/sweep_erasure_dilation.py --dilations 8 6 4 3 2
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

from pidetect.graph.build import (
    run_step5, run_step6, build_sr_sc, veto_short_gap_by_crossing_separation,
)
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, run_step9
from pidetect.graph.lines import binarize as _binarize_orig

from run_step4_veto_eval import _build_through_s4
from decompose_p3 import _corridor_continuity

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]
DEFAULT_DILATIONS = [8, 6, 4, 3, 2]

MECH_ORDER = [
    "P3c(contracted:connector)", "P3c(contracted:crossing)",
    "P3b(bbox_touch_no_ink)", "P3b(third_node_ink)",
    "P3ab(short_gap_ink)", "P3ab(skeleton_branch)",
]


def _classify_fp(gt_u, gt_v, gt_G, violation_set) -> str:
    if frozenset((gt_u, gt_v)) in violation_set:
        return "S4"
    if not nx.has_path(gt_G, gt_u, gt_v):
        return "P3"
    return "OTHER"


def _short_bound_stub_count(all_symbol_nodes, s3, min_branch_len_px: float) -> int:
    """Skeleton branches < 2*min_branch_len_px with >=1 endpoint bound to a
    valve/instrument graph node -- the measurable "new spurious stub" proxy."""
    scored_ids = {
        n.node_id for n in all_symbol_nodes
        if n.is_graph_node and n.node_type in ("valve", "instrument")
    }
    bound_branch_ids = {
        ep.branch_id for ep in s3.endpoints
        if ep.bound_node_id is not None and ep.bound_node_id in scored_ids
    }
    thresh = 2.0 * min_branch_len_px
    n = 0
    for b in s3.branches:
        if b.branch_id in bound_branch_ids and b.length_px < thresh:
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dilations", nargs="+", type=int, default=DEFAULT_DILATIONS)
    args = p.parse_args()

    with open(CFG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    weights = REPO / args.weights
    cache_dir = REPO / args.cache_dir
    baseline_dilation = args.dilations[0]

    all_results: dict[int, dict[int, dict]] = {}  # dilation -> sheet -> result dict

    for dilation in args.dilations:
        cfg = dict(base_cfg)
        cfg["erase_dilation_px"] = dilation
        print(f"\n{'#'*90}\n# erase_dilation_px = {dilation}\n{'#'*90}")

        sheet_results: dict[int, dict] = {}
        for sheet_id in SHEETS:
            png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
                sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
            )
            SR, Sc = build_sr_sc(all_symbol_nodes, s3.off_page_nodes, s3, s4)
            kept, suppressed = veto_short_gap_by_crossing_separation(
                s3.short_gap_pairs, SR, Sc
            )
            s3_final = dataclasses.replace(s3, short_gap_pairs=kept)
            s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3_final, s4)
            s6 = run_step6(s5)
            s9 = run_step9(s6.graph, gml)

            gt_G, crossing_infos = load_gt_contracted(gml)
            pred_to_gt, gt_to_pred = match_nodes(s6.graph, gt_G)
            gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
            violation_set = {
                frozenset((a, b)) for info in crossing_infos for a, b in info["separated_pairs"]
            }

            node_by_id = {n.node_id: n for n in all_symbol_nodes + s3.off_page_nodes}
            img_bgr = cv2.imread(str(png))
            _, binary_pre_erase = _binarize_orig(
                img_bgr, blur_sigma=cfg["blur_sigma"], close_disk_r=0, open_disk_r=0
            )

            fp_by_mech = {m: 0 for m in MECH_ORDER}
            fp_pair_ids: set[frozenset] = set()
            tp_pair_ids: set[frozenset] = set()
            shared_header_pairs: set[frozenset] = set()

            for u, v, edata in s6.graph.edges(data=True):
                gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
                if gt_u is None or gt_v is None:
                    continue
                id_a, id_b = int(u.split("_")[1]), int(v.split("_")[1])
                pair_key = frozenset((id_a, id_b))
                if frozenset((gt_u, gt_v)) in gt_edge_set:
                    tp_pair_ids.add(pair_key)
                    continue
                bucket = _classify_fp(gt_u, gt_v, gt_G, violation_set)
                if bucket != "P3":
                    continue

                is_sg = bool(edata.get("short_gap", False))
                is_contracted = bool(edata.get("contracted", False))
                via_type = edata.get("via_type")
                a_node, b_node = node_by_id.get(id_a), node_by_id.get(id_b)

                if is_contracted:
                    mech = f"P3c(contracted:{via_type})"
                elif is_sg and a_node is not None and b_node is not None:
                    _cont_frac, _n_dark, third_frac = _corridor_continuity(
                        binary_pre_erase, a_node, b_node, all_symbol_nodes
                    )
                    orig_gx = max(0.0, a_node.x1 - b_node.x2, b_node.x1 - a_node.x2)
                    orig_gy = max(0.0, a_node.y1 - b_node.y2, b_node.y1 - a_node.y2)
                    bbox_touch = (orig_gx == 0.0 and orig_gy == 0.0)
                    if bbox_touch:
                        mech = "P3b(bbox_touch_no_ink)"
                    elif third_frac >= 0.5:
                        mech = "P3b(third_node_ink)"
                    else:
                        mech = "P3ab(short_gap_ink)"
                        shared_header_pairs.add(pair_key)
                else:
                    mech = "P3ab(skeleton_branch)"

                fp_by_mech[mech] = fp_by_mech.get(mech, 0) + 1
                fp_pair_ids.add(pair_key)

            n_short_stubs = _short_bound_stub_count(
                all_symbol_nodes, s3, cfg["min_branch_len_px"]
            )

            sheet_results[sheet_id] = dict(
                s9=s9, fp_by_mech=fp_by_mech, fp_pair_ids=fp_pair_ids,
                tp_pair_ids=tp_pair_ids, shared_header_pairs=shared_header_pairs,
                n_short_stubs=n_short_stubs,
                n_connectors=s4.n_connectors, n_crossings=s4.n_crossings,
                n_branches=len(s3.branches),
            )
            print(f"  sheet {sheet_id}: TP={s9.edge_tp} FP={s9.edge_fp} FN={s9.edge_fn} "
                  f"F1={s9.edge_f1:.3f}  P3ab(short_gap_ink)={fp_by_mech['P3ab(short_gap_ink)']}  "
                  f"short_bound_stubs={n_short_stubs}  connectors={s4.n_connectors} crossings={s4.n_crossings}")

        all_results[dilation] = sheet_results

    # ------------------------------------------------------------------
    # SWEEP TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SWEEP TABLE")
    print("=" * 100)
    header = (f"{'dil':<5}{'sheet':<7}{'TP':<5}{'FP':<5}{'FN':<5}{'P':<8}{'R':<8}{'F1':<8}"
              f"{'shared_hdr_FP':<15}{'short_stubs':<13}{'connectors':<12}{'crossings':<10}")
    print(header)
    for dilation in args.dilations:
        tp_sum = fp_sum = fn_sum = sh_sum = 0
        for sheet_id in SHEETS:
            r = all_results[dilation][sheet_id]
            s9 = r["s9"]
            tp_sum += s9.edge_tp; fp_sum += s9.edge_fp; fn_sum += s9.edge_fn
            sh_sum += r["fp_by_mech"]["P3ab(short_gap_ink)"]
            print(f"{dilation:<5}{sheet_id:<7}{s9.edge_tp:<5}{s9.edge_fp:<5}{s9.edge_fn:<5}"
                  f"{s9.edge_precision:<8.3f}{s9.edge_recall:<8.3f}{s9.edge_f1:<8.3f}"
                  f"{r['fp_by_mech']['P3ab(short_gap_ink)']:<15}{r['n_short_stubs']:<13}"
                  f"{r['n_connectors']:<12}{r['n_crossings']:<10}")
        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        print(f"{dilation:<5}{'MEAN':<7}{tp_sum:<5}{fp_sum:<5}{fn_sum:<5}"
              f"{p:<8.3f}{rec:<8.3f}{f1:<8.3f}{sh_sum:<15}")
        print("-" * len(header))

    # ------------------------------------------------------------------
    # TP DIFF vs baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"TP DIFF vs baseline (dilation={baseline_dilation})")
    print("=" * 100)
    for dilation in args.dilations:
        if dilation == baseline_dilation:
            continue
        any_change = False
        for sheet_id in SHEETS:
            base_tp = all_results[baseline_dilation][sheet_id]["tp_pair_ids"]
            cur_tp = all_results[dilation][sheet_id]["tp_pair_ids"]
            lost = base_tp - cur_tp
            gained = cur_tp - base_tp
            if lost or gained:
                any_change = True
                print(f"  dilation={dilation} sheet={sheet_id}: "
                      f"lost={sorted(tuple(sorted(x)) for x in lost)} "
                      f"gained={sorted(tuple(sorted(x)) for x in gained)}")
        if not any_change:
            print(f"  dilation={dilation}: TP set IDENTICAL to baseline on all 3 sheets "
                  f"(counts: {[len(all_results[dilation][s]['tp_pair_ids']) for s in SHEETS]})")

    # ------------------------------------------------------------------
    # Shared-header FP pairs KILLED vs baseline (present as FP at baseline,
    # gone entirely -- not just re-classified -- at this dilation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SHARED-HEADER FP PAIRS KILLED vs baseline")
    print("=" * 100)
    for dilation in args.dilations:
        if dilation == baseline_dilation:
            continue
        for sheet_id in SHEETS:
            base_sh = all_results[baseline_dilation][sheet_id]["shared_header_pairs"]
            cur_fp = all_results[dilation][sheet_id]["fp_pair_ids"]
            killed = base_sh - cur_fp
            still_fp = base_sh & cur_fp
            print(f"  dilation={dilation} sheet={sheet_id}: "
                  f"baseline_shared_header={len(base_sh)} killed={len(killed)} still_FP={len(still_fp)}")

    print("\n" + "=" * 100)
    print(f"SHORT-BOUND-STUB DELTA vs baseline (dilation={baseline_dilation}) -- proxy for new spurious stubs")
    print("=" * 100)
    for dilation in args.dilations:
        deltas = []
        for sheet_id in SHEETS:
            base_n = all_results[baseline_dilation][sheet_id]["n_short_stubs"]
            cur_n = all_results[dilation][sheet_id]["n_short_stubs"]
            deltas.append(cur_n - base_n)
        print(f"  dilation={dilation}: delta per sheet {deltas}  total_delta={sum(deltas)}")


if __name__ == "__main__":
    main()
