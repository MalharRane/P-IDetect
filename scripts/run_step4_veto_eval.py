"""Phase 4 step 4 — SR/Sc crossing-separation veto: TP-safety gate + evaluation.

Implements the full validation protocol requested for the step-4 implementation:

  1. TP-safety gate (extends docs/phase4_step4_scope.md §6c from sheet-3-only,
     short-gap-TPs-only to ALL scoreable TP edges on ALL THREE sheets). For every
     TP edge, computes sr_conn / sc_conn against SR/Sc built from the sheet's
     ORIGINAL (pre-veto) short_gap_pairs, and whether the veto would fire on it.
     Only short-gap TPs can ever be vetoed (skeleton-traced edges are never in
     short_gap_pairs), but every TP is checked and printed for transparency.

  2. If ANY TP shows sr_conn=True and sc_conn=False (veto would delete a real
     edge): STOP. Print the SR path, identify the crossing whose pass-through
     choice breaks it, and crop the sheet image around that crossing so the
     false-positive/genuine-crossing question can be inspected visually.

  3. Only if every TP on every sheet is safe: apply the veto, recompute
     TP/FP/FN/P/R/F1 for all three sheets, and report deltas against the frozen
     step-3 baseline (docs/phase4_step3_final.md), the sheet-10 control check,
     and confirmation that the 8 Population-B (skeleton-traced) S4 FPs are
     untouched.

Usage:
    PYTHONPATH=src python scripts/run_step4_veto_eval.py --device cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import networkx as nx
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import build_node_set, centroid_nms, erase_symbols, roi_filter
from pidetect.graph.lines import binarize as _binarize_orig, run_step3, assert_run_step3_wired
from pidetect.graph.junction import run_step4
from pidetect.graph.build import (
    run_step5, run_step6, sym_gid,
    build_sr_sc, veto_short_gap_by_crossing_separation,
)
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, edge_metrics, run_step9

from run_phase4_steps03 import _load_bg_bboxes, _build_vessel_nodes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
OUT_DIR = REPO / "docs" / "phase4_step4_design"
SHEETS = [0, 3, 10]

# Frozen step-3 baseline (docs/phase4_step3_final.md, "Three-Sheet Metrics" table)
FROZEN = {
    0:  dict(tp=19, fp=34, fn=12, p=0.358, r=0.613, f1=0.452),
    3:  dict(tp=8,  fp=13, fn=2,  p=0.381, r=0.800, f1=0.516),
    10: dict(tp=14, fp=17, fn=8,  p=0.452, r=0.636, f1=0.528),
}

# Population A -- short-gap S4 FPs the veto targets (docs/phase4_step4_scope.md §4)
POP_A: dict[int, list[tuple[int, int]]] = {
    0: [
        (10, 31), (12, 30), (12, 80), (14, 33), (14, 88), (8, 27), (8, 82),
        (20, 21), (20, 35), (20, 107), (21, 23), (21, 87), (21, 104),
        (22, 35), (22, 101), (23, 28), (23, 107), (23, 115),
        (35, 98), (35, 113), (36, 105), (87, 115), (92, 115),
        (101, 103), (105, 113),
    ],
    3: [(12, 54), (5, 54), (54, 74)],
    10: [(155, 185), (159, 177), (159, 190)],
}

# Population B -- skeleton-traced S4 FPs, NOT targeted (should remain untouched)
POP_B: dict[int, list[tuple[int, int]]] = {
    0: [(21, 35), (30, 36)],
    3: [(0, 1), (0, 11), (0, 74), (1, 11), (1, 74), (11, 74)],
    10: [],
}


def _pair_key(a: int, b: int) -> frozenset:
    return frozenset((a, b))


def _build_through_s4(sheet_id: int, weights: Path, cache_dir: Path, conf: float,
                       iou: float, device: str, cfg: dict):
    """Run steps 0-4 for one sheet, returning everything needed downstream."""
    png = RAW_DIR / f"{sheet_id}.png"
    gml = RAW_DIR / f"{sheet_id}.graphml"

    bg_bboxes, img_w, img_h, vessel_entries = _load_bg_bboxes(gml)
    raw_preds = _load_or_infer(sheet_id, png, weights, conf, iou, device, cache_dir)

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


def _diagnose_risk(SR: nx.Graph, gid_a: str, gid_b: str) -> tuple[list[str], list[tuple[str, float, float]]]:
    """Find the SR shortest path A->B and any crossing nodes on it whose
    pass-through pairing does not match the branches actually used by the path."""
    path = nx.shortest_path(SR, gid_a, gid_b)
    culprits: list[tuple[str, float, float]] = []
    for i in range(1, len(path) - 1):
        node = path[i]
        d = SR.nodes[node]
        if d.get("node_type") != "crossing":
            continue
        bid_prev = SR.edges[path[i - 1], node].get("branch_id")
        bid_next = SR.edges[node, path[i + 1]].get("branch_id")
        pass_pairs = d.get("pass_through_pairs", [])
        matched = any({bid_prev, bid_next} == {a, b} for a, b in pass_pairs)
        if not matched:
            culprits.append((node, d.get("cx", 0.0), d.get("cy", 0.0)))
    return path, culprits


def _crop_and_save(png: Path, cx: float, cy: float, sheet_id: int, junc_gid: str, margin: int = 250) -> Path:
    with Image.open(png) as im:
        w, h = im.size
        box = (max(0, int(cx - margin)), max(0, int(cy - margin)),
               min(w, int(cx + margin)), min(h, int(cy + margin)))
        crop = im.crop(box)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"sheet{sheet_id}_risk_{junc_gid}.png"
        crop.save(out_path)
    return out_path


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

    per_sheet = {}
    safety_rows: list[tuple[int, str, str, bool, bool, bool, bool]] = []
    # (sheet, gid_a, gid_b, is_short_gap, sr_conn, sc_conn, veto_would_fire)
    risks: list[tuple[int, str, str]] = []

    for sheet_id in SHEETS:
        png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
        )

        # Baseline (no veto) -- current code's production graph, s4 contraction applied
        # but no short-gap veto. Compared against both FROZEN (drift check) and the
        # post-veto numbers (isolates the veto's own effect).
        s5_base = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        s6_base = run_step6(s5_base)
        s9_base = run_step9(s6_base.graph, gml)

        gt_G, crossing_infos = load_gt_contracted(gml)
        pred_to_gt, gt_to_pred = match_nodes(s6_base.graph, gt_G)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}

        # SR / Sc built from the ORIGINAL (pre-veto) s3 -- these are what the veto
        # would actually see when deciding whether to suppress each short-gap pair.
        SR, Sc = build_sr_sc(all_symbol_nodes, s3.off_page_nodes, s3, s4)

        # Every TP edge in the current (pre-veto) graph.
        for u, v, edata in s6_base.graph.edges(data=True):
            gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
            if gt_u is None or gt_v is None:
                continue
            if frozenset((gt_u, gt_v)) not in gt_edge_set:
                continue  # not a TP
            is_sg = bool(edata.get("short_gap", False))
            sr_conn = u in SR and v in SR and nx.has_path(SR, u, v)
            sc_conn = u in Sc and v in Sc and nx.has_path(Sc, u, v)
            veto_fires = is_sg and sr_conn and not sc_conn
            safety_rows.append((sheet_id, u, v, is_sg, sr_conn, sc_conn, veto_fires))
            if veto_fires:
                risks.append((sheet_id, u, v))

        per_sheet[sheet_id] = dict(
            png=png, gml=gml, all_symbol_nodes=all_symbol_nodes,
            s3=s3, s4=s4, SR=SR, Sc=Sc, s9_base=s9_base,
        )

        # Diagnostic: sr_conn/sc_conn for the DOCUMENTED Population-A pairs, regardless
        # of whether the veto actually suppressed them. Explains a 0-suppression result.
        sg_set = {_pair_key(a, b) for a, b in s3.short_gap_pairs}
        pop_a_diag = []
        for a, b in POP_A[sheet_id]:
            ga, gb = sym_gid(a), sym_gid(b)
            still_candidate = _pair_key(a, b) in sg_set
            sr_conn = ga in SR and gb in SR and nx.has_path(SR, ga, gb)
            sc_conn = ga in Sc and gb in Sc and nx.has_path(Sc, ga, gb)
            pop_a_diag.append((a, b, still_candidate, sr_conn, sc_conn))
        per_sheet[sheet_id]["pop_a_diag"] = pop_a_diag

    # ------------------------------------------------------------------
    # 1. Print the TP-safety table (all sheets, all scoreable TPs)
    # ------------------------------------------------------------------
    print("=" * 78)
    print("TP-SAFETY TABLE -- all scoreable TP edges, sheets 0/3/10")
    print("=" * 78)
    print(f"{'sheet':<7}{'pair':<24}{'short_gap?':<12}{'sr_conn':<10}{'sc_conn':<10}{'veto_fires?'}")
    for sheet_id, u, v, is_sg, sr_conn, sc_conn, veto_fires in sorted(safety_rows):
        pair_str = f"{u}<->{v}"
        print(f"{sheet_id:<7}{pair_str:<24}{str(is_sg):<12}{str(sr_conn):<10}{str(sc_conn):<10}{veto_fires}")

    n_tp_total = len(safety_rows)
    n_sg_tp = sum(1 for r in safety_rows if r[3])
    print(f"\nTotal TPs checked: {n_tp_total}  (short-gap TPs: {n_sg_tp})")

    # ------------------------------------------------------------------
    # 2. STOP if any risk
    # ------------------------------------------------------------------
    if risks:
        print("\n" + "!" * 78)
        print(f"GATE FAILED: {len(risks)} TP(s) would be wrongly suppressed by the veto.")
        print("!" * 78)
        for sheet_id, gid_a, gid_b in risks:
            info = per_sheet[sheet_id]
            SR = info["SR"]
            path, culprits = _diagnose_risk(SR, gid_a, gid_b)
            print(f"\nSheet {sheet_id}: {gid_a} <-> {gid_b}")
            print(f"  SR path: {' -> '.join(path)}")
            if not culprits:
                print("  No single crossing on the shortest path explains the break "
                      "(disconnection likely arises from global contraction interactions; "
                      "manual inspection required).")
            for junc_gid, cx, cy in culprits:
                print(f"  Responsible crossing: {junc_gid} at (cx={cx:.0f}, cy={cy:.0f})")
                crop_path = _crop_and_save(info["png"], cx, cy, sheet_id, junc_gid)
                print(f"  Crop saved -> {crop_path}")
        print("\nSTOPPING before FP/precision reporting per instructions.")
        return

    print("\nGATE PASSED: all TPs across all three sheets are safe "
          "(no TP shows sr_conn=True and sc_conn=False).")

    # ------------------------------------------------------------------
    # 2b. Baseline (no veto, current code) vs FROZEN drift check
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BASELINE (current code, NO veto) vs FROZEN step-3 record -- drift check")
    print("=" * 78)
    print(f"{'sheet':<7}{'TP':<6}{'FP':<6}{'FN':<6}{'P':<8}{'R':<8}{'F1':<8}{'(frozen P/R/F1)'}")
    for sheet_id in SHEETS:
        b = per_sheet[sheet_id]["s9_base"]
        fz = FROZEN[sheet_id]
        print(f"{sheet_id:<7}{b.edge_tp:<6}{b.edge_fp:<6}{b.edge_fn:<6}"
              f"{b.edge_precision:<8.3f}{b.edge_recall:<8.3f}{b.edge_f1:<8.3f}"
              f"({fz['p']:.3f}/{fz['r']:.3f}/{fz['f1']:.3f})")
        if (b.edge_tp, b.edge_fp, b.edge_fn) != (fz["tp"], fz["fp"], fz["fn"]):
            print(f"  NOTE: current no-veto baseline differs from the frozen record -- "
                  f"the delta below vs. 'frozen' includes this drift, not just the veto's effect.")

    # ------------------------------------------------------------------
    # 3. Apply veto, recompute metrics, report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("APPLYING VETO -- recomputing TP/FP/FN/P/R/F1 per sheet")
    print("=" * 78)

    new_results = {}
    for sheet_id in SHEETS:
        info = per_sheet[sheet_id]
        s3, s4 = info["s3"], info["s4"]
        SR, Sc = info["SR"], info["Sc"]
        all_symbol_nodes = info["all_symbol_nodes"]
        gml = info["gml"]

        kept_pairs, suppressed_pairs = veto_short_gap_by_crossing_separation(
            s3.short_gap_pairs, SR, Sc
        )
        s3_vetoed = dataclasses.replace(s3, short_gap_pairs=kept_pairs)
        s5_new = run_step5(all_symbol_nodes, s3.off_page_nodes, s3_vetoed, s4)
        s6_new = run_step6(s5_new)
        s9_new = run_step9(s6_new.graph, gml)

        expected_a = {_pair_key(a, b) for a, b in POP_A[sheet_id]}
        suppressed_keys = {_pair_key(a, b) for a, b in suppressed_pairs}
        n_a_suppressed = len(expected_a & suppressed_keys)
        extra_suppressed = suppressed_keys - expected_a

        # Confirm Population-B pairs remain as (unmatched / FP) edges in the new graph.
        gid_to_id = {}
        for n in all_symbol_nodes:
            gid_to_id[sym_gid(n.node_id)] = n.node_id
        pop_b_untouched = []
        for a, b in POP_B[sheet_id]:
            ga, gb = sym_gid(a), sym_gid(b)
            still_present = s6_new.graph.has_edge(ga, gb)
            pop_b_untouched.append((a, b, still_present))

        new_results[sheet_id] = dict(
            suppressed_pairs=suppressed_pairs,
            n_a_suppressed=n_a_suppressed,
            n_a_expected=len(expected_a),
            extra_suppressed=extra_suppressed,
            pop_b_untouched=pop_b_untouched,
            s9=s9_new,
        )

    print(f"\n{'sheet':<7}{'total supp.':<13}{'SG-FP(PopA) supp.':<20}{'TP':<6}{'FP':<6}{'FN':<6}"
          f"{'P':<8}{'R':<8}{'F1':<8}{'(vs baseline TP/FP/FN)'}")
    for sheet_id in SHEETS:
        r = new_results[sheet_id]
        s9 = r["s9"]
        b = per_sheet[sheet_id]["s9_base"]
        supp_str = f"{r['n_a_suppressed']}/{r['n_a_expected']}"
        total_supp = len(r["suppressed_pairs"])
        print(f"{sheet_id:<7}{total_supp:<13}{supp_str:<20}{s9.edge_tp:<6}{s9.edge_fp:<6}{s9.edge_fn:<6}"
              f"{s9.edge_precision:<8.3f}{s9.edge_recall:<8.3f}{s9.edge_f1:<8.3f}"
              f"({b.edge_tp}/{b.edge_fp}/{b.edge_fn})")
        if r["suppressed_pairs"]:
            print(f"  Suppressed pairs (raw ids): {sorted(r['suppressed_pairs'])}")
        if r["extra_suppressed"]:
            print(f"  NOTE: {len(r['extra_suppressed'])} suppressed pair(s) outside the "
                  f"documented Population-A list: {sorted(r['extra_suppressed'])}")

    mean_p = sum(new_results[s]["s9"].edge_precision for s in SHEETS) / 3
    mean_r = sum(new_results[s]["s9"].edge_recall for s in SHEETS) / 3
    mean_f1 = sum(new_results[s]["s9"].edge_f1 for s in SHEETS) / 3
    base_mean_p = sum(per_sheet[s]["s9_base"].edge_precision for s in SHEETS) / 3
    base_mean_r = sum(per_sheet[s]["s9_base"].edge_recall for s in SHEETS) / 3
    base_mean_f1 = sum(per_sheet[s]["s9_base"].edge_f1 for s in SHEETS) / 3
    print(f"\nmean (with veto):    P={mean_p:.3f} R={mean_r:.3f} F1={mean_f1:.3f}")
    print(f"mean (no-veto base): P={base_mean_p:.3f} R={base_mean_r:.3f} F1={base_mean_f1:.3f}")
    print(f"mean (frozen record):P=0.397 R=0.683 F1=0.499")

    # Sheet-10 control -- compare against the CURRENT no-veto baseline (same code,
    # same config, only the veto differs) so the check isolates the veto's effect
    # rather than any drift between today's pipeline and the frozen record.
    r10 = new_results[10]
    b10 = per_sheet[10]["s9_base"]
    print("\n--- Sheet 10 control ---")
    print(f"  Recall: {b10.edge_recall:.4f} (no-veto baseline) -> {r10['s9'].edge_recall:.4f} (with veto)")
    print(f"  FP:     {b10.edge_fp} (no-veto baseline) -> {r10['s9'].edge_fp} (with veto)  "
          f"(SG-FPs suppressed: {r10['n_a_suppressed']}/{r10['n_a_expected']}, "
          f"total suppressed: {len(r10['suppressed_pairs'])})")
    if b10.edge_tp != r10["s9"].edge_tp or b10.edge_fn != r10["s9"].edge_fn:
        print("  RED FLAG: sheet-10 TP/FN changed relative to the no-veto baseline -- "
              "the veto is suppressing a TP despite the safety gate passing.")
    else:
        print("  OK: TP/FN unchanged relative to the no-veto baseline "
              "(recall necessarily unchanged too).")

    # Population B confirmation
    print("\n--- Population B (skeleton-traced S4 FPs) untouched check ---")
    for sheet_id in SHEETS:
        for a, b, present in new_results[sheet_id]["pop_b_untouched"]:
            status = "still present (untouched, as expected)" if present else "MISSING (unexpectedly removed!)"
            print(f"  sheet {sheet_id}: (sym_{a},sym_{b}) -> {status}")

    print("\n--- Population-A documented target pairs: why 0 suppressed? ---")
    print(f"{'sheet':<7}{'pair':<20}{'still_candidate?':<18}{'sr_conn':<10}{'sc_conn':<10}")
    for sheet_id in SHEETS:
        for a, b, still_candidate, sr_conn, sc_conn in per_sheet[sheet_id]["pop_a_diag"]:
            pair_str = f"sym_{a}<->sym_{b}"
            print(f"{sheet_id:<7}{pair_str:<20}{str(still_candidate):<18}{str(sr_conn):<10}{str(sc_conn)}")


if __name__ == "__main__":
    main()
