"""Phase 4 — re-baseline against CURRENT production output (measurement only).

docs/phase4_step3_final.md was frozen before run_step4 crossing/connector
contraction was wired into build.py's step5/step6 graph construction. That
wiring already changes TP/FP/FN relative to the frozen record, independent of
the SR/Sc veto (which is a currently-measured no-op — see run_step4_veto_eval.py).
This script re-measures everything against TODAY's contracted graph:

  1. Per-sheet TP/FP/FN, P/R/F1, node-match %, crossing non-edge % -- the new
     true baseline (production pipeline, veto included, though it fires 0x).
  2. Re-classifies every current FP into S4-topology / P3-signal-line / OTHER
     against today's GT-contracted graph (see _classify_fp for the exact
     operational definition).
  3. Diffs today's FP pair set against the frozen FP list per sheet: which
     frozen FPs are gone (and why), which are new.
  4. States whether the SR/Sc veto still has any target population at all on
     today's output (short-gap FPs with sr_conn=True and sc_conn=False).
  5. Traces the sym_10<->sym_31 (sheet 0) anomaly: SR path, any junction nodes
     on it, and crops the image around them for visual inspection.

No mechanism changes. Usage:
    PYTHONPATH=src python scripts/rebaseline_step4.py --device cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import networkx as nx
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.build import (
    run_step5, run_step6, sym_gid,
    build_sr_sc, veto_short_gap_by_crossing_separation,
)
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, run_step9

from run_step4_veto_eval import _build_through_s4, _pair_key, _diagnose_risk, _crop_and_save

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
OUT_DIR = REPO / "docs" / "phase4_step4_design"
SHEETS = [0, 3, 10]

FROZEN = {
    0:  dict(tp=19, fp=34, fn=12, p=0.358, r=0.613, f1=0.452),
    3:  dict(tp=8,  fp=13, fn=2,  p=0.381, r=0.800, f1=0.516),
    10: dict(tp=14, fp=17, fn=8,  p=0.452, r=0.636, f1=0.528),
}

# Full frozen FP detail tables from docs/phase4_step3_final.md ("FP Detail (per sheet)").
# (id_a, id_b, bucket, short_gap?)
FROZEN_FP: dict[int, list[tuple[int, int, str, bool]]] = {
    0: [
        (111, 112, "OTHER", True),
        (105, 108, "P3", True), (108, 113, "P3", True), (29, 34, "P3", True),
        (29, 84, "P3", True), (37, 100, "P3", True), (84, 85, "P3", True),
        (10, 31, "S4", True), (101, 103, "S4", True), (105, 113, "S4", True),
        (12, 30, "S4", True), (12, 80, "S4", True), (14, 33, "S4", True),
        (14, 88, "S4", True), (20, 21, "S4", True), (20, 35, "S4", True),
        (20, 107, "S4", True), (21, 23, "S4", True), (21, 87, "S4", True),
        (21, 104, "S4", True), (21, 35, "S4", False), (22, 35, "S4", True),
        (22, 101, "S4", True), (23, 28, "S4", True), (23, 107, "S4", True),
        (23, 115, "S4", True), (30, 36, "S4", False), (35, 98, "S4", True),
        (35, 113, "S4", True), (36, 105, "S4", True), (8, 27, "S4", True),
        (8, 82, "S4", True), (87, 115, "S4", True), (92, 115, "S4", True),
    ],
    3: [
        (10, 66, "OTHER", False),
        (2, 59, "P3", True), (3, 66, "P3", True), (9, 56, "P3", True),
        (0, 11, "S4", False), (0, 1, "S4", False), (0, 74, "S4", False),
        (1, 11, "S4", False), (1, 74, "S4", False), (11, 74, "S4", False),
        (12, 54, "S4", True), (5, 54, "S4", True), (54, 74, "S4", True),
    ],
    10: [
        (162, 189, "OTHER", False), (33, 50, "OTHER", True), (34, 45, "OTHER", True),
        (38, 162, "OTHER", False),
        (159, 160, "P3", True), (159, 178, "P3", True), (160, 177, "P3", True),
        (160, 178, "P3", True), (160, 190, "P3", True), (164, 167, "P3", True),
        (169, 183, "P3", True), (177, 178, "P3", True), (5, 19, "P3", True),
        (5, 162, "P3", True),
        (155, 185, "S4", True), (159, 177, "S4", True), (159, 190, "S4", True),
    ],
}


def _classify_fp(gt_u: str, gt_v: str, gt_G: nx.Graph, violation_set: set[frozenset]) -> str:
    """S4 = GT crossing-separation violation. P3 = no GT path at all. OTHER = rest."""
    if frozenset((gt_u, gt_v)) in violation_set:
        return "S4"
    if not nx.has_path(gt_G, gt_u, gt_v):
        return "P3"
    return "OTHER"


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

    results = {}

    for sheet_id in SHEETS:
        png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
        )

        # Production wiring: build SR/Sc, apply veto (no-op today), build final graph.
        SR, Sc = build_sr_sc(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        kept_pairs, suppressed_pairs = veto_short_gap_by_crossing_separation(
            s3.short_gap_pairs, SR, Sc
        )
        s3_final = dataclasses.replace(s3, short_gap_pairs=kept_pairs)
        s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3_final, s4)
        s6 = run_step6(s5)
        s9 = run_step9(s6.graph, gml)

        gt_G, crossing_infos = load_gt_contracted(gml)
        pred_to_gt, gt_to_pred = match_nodes(s6.graph, gt_G)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
        violation_set = {
            frozenset((a, b)) for info in crossing_infos for a, b in info["separated_pairs"]
        }

        # Classify every current FP edge.
        fp_rows = []  # (id_a, id_b, bucket, is_short_gap)
        for u, v, edata in s6.graph.edges(data=True):
            gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
            if gt_u is None or gt_v is None:
                continue
            if frozenset((gt_u, gt_v)) in gt_edge_set:
                continue  # TP
            bucket = _classify_fp(gt_u, gt_v, gt_G, violation_set)
            is_sg = bool(edata.get("short_gap", False))
            id_a, id_b = int(u.split("_")[1]), int(v.split("_")[1])
            fp_rows.append((id_a, id_b, bucket, is_sg))

        # Short-gap FPs where the veto COULD have fired (sr=True, sc=False) -- checked
        # directly on today's SR/Sc, over every current short-gap pair (FP or TP).
        veto_targets = []
        for id_a, id_b in s3.short_gap_pairs:
            ga, gb = sym_gid(id_a), sym_gid(id_b)
            sr_conn = ga in SR and gb in SR and nx.has_path(SR, ga, gb)
            sc_conn = ga in Sc and gb in Sc and nx.has_path(Sc, ga, gb)
            if sr_conn and not sc_conn:
                veto_targets.append((id_a, id_b))

        results[sheet_id] = dict(
            png=png, gml=gml, s3=s3, s4=s4, SR=SR, Sc=Sc,
            s9=s9, fp_rows=fp_rows, veto_targets=veto_targets,
            all_symbol_nodes=all_symbol_nodes, s6=s6,
            gt_G=gt_G, pred_to_gt=pred_to_gt, gt_edge_set=gt_edge_set,
            violation_set=violation_set,
        )

    # ------------------------------------------------------------------
    # 1. New baseline table
    # ------------------------------------------------------------------
    print("=" * 90)
    print("NEW BASELINE -- current production output (run_step4 contraction wired, veto no-op)")
    print("=" * 90)
    print(f"{'sheet':<7}{'match%':<9}{'TP':<5}{'FP':<5}{'FN':<5}{'P':<8}{'R':<8}{'F1':<8}"
          f"{'cross%':<9}{'(frozen P/R/F1)'}")
    for sheet_id in SHEETS:
        s9 = results[sheet_id]["s9"]
        fz = FROZEN[sheet_id]
        print(f"{sheet_id:<7}{s9.match_recall:<9.1%}{s9.edge_tp:<5}{s9.edge_fp:<5}{s9.edge_fn:<5}"
              f"{s9.edge_precision:<8.3f}{s9.edge_recall:<8.3f}{s9.edge_f1:<8.3f}"
              f"{s9.crossing_nonedge_rate:<9.1%}({fz['p']:.3f}/{fz['r']:.3f}/{fz['f1']:.3f})")
    mean_match = sum(results[s]["s9"].match_recall for s in SHEETS) / 3
    mean_p = sum(results[s]["s9"].edge_precision for s in SHEETS) / 3
    mean_r = sum(results[s]["s9"].edge_recall for s in SHEETS) / 3
    mean_f1 = sum(results[s]["s9"].edge_f1 for s in SHEETS) / 3
    mean_cross = sum(results[s]["s9"].crossing_nonedge_rate for s in SHEETS) / 3
    print(f"{'mean':<7}{mean_match:<9.1%}{'':<15}{mean_p:<8.3f}{mean_r:<8.3f}{mean_f1:<8.3f}"
          f"{mean_cross:<9.1%}(0.397/0.683/0.499)")

    # ------------------------------------------------------------------
    # 2. Re-classified FP buckets
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RE-CLASSIFIED FP BUCKETS (today's contracted graph)")
    print("=" * 90)
    print(f"{'sheet':<7}{'total FP':<10}{'S4':<14}{'P3':<14}{'OTHER':<14}")
    totals = {"S4": 0, "P3": 0, "OTHER": 0}
    for sheet_id in SHEETS:
        rows = results[sheet_id]["fp_rows"]
        n = len(rows)
        counts = {"S4": 0, "P3": 0, "OTHER": 0}
        for _, _, bucket, _ in rows:
            counts[bucket] += 1
            totals[bucket] += 1
        def fmt(k):
            c = counts[k]
            pct = 100 * c / n if n else 0.0
            return f"{c} ({pct:.0f}%)"
        print(f"{sheet_id:<7}{n:<10}{fmt('S4'):<14}{fmt('P3'):<14}{fmt('OTHER'):<14}")
    grand_total = sum(totals.values())
    def fmt_total(k):
        c = totals[k]
        pct = 100 * c / grand_total if grand_total else 0.0
        return f"{c} ({pct:.0f}%)"
    print(f"{'total':<7}{grand_total:<10}{fmt_total('S4'):<14}{fmt_total('P3'):<14}{fmt_total('OTHER'):<14}")

    print("\nFrozen record composition (docs/phase4_step3_final.md), for comparison:")
    print("  sheet 0: total 34, S4 27 (79%), P3 6 (17%), OTHER 1 (2%)")
    print("  sheet 3: total 13, S4 9 (69%), P3 3 (23%), OTHER 1 (7%)")
    print("  sheet 10: total 17, S4 3 (17%), P3 10 (58%), OTHER 4 (23%)")
    print("  total: 64, S4 39 (60%), P3 19 (29%), OTHER 6 (9%)")

    # ------------------------------------------------------------------
    # 3. Diff vs frozen FP list
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DIFF vs FROZEN FP LIST -- what did already-wired contraction change?")
    print("=" * 90)
    for sheet_id in SHEETS:
        s6_fp_set = {_pair_key(a, b) for a, b, _, _ in results[sheet_id]["fp_rows"]}
        frozen_set = {_pair_key(a, b) for a, b, _, _ in FROZEN_FP[sheet_id]}
        gone = frozen_set - s6_fp_set
        new = s6_fp_set - frozen_set
        unchanged = frozen_set & s6_fp_set
        print(f"\nSheet {sheet_id}: frozen={len(frozen_set)} today={len(s6_fp_set)} "
              f"unchanged={len(unchanged)} gone={len(gone)} new={len(new)}")
        if gone:
            print(f"  Gone (no longer an FP today): {sorted(gone)}")
        if new:
            print(f"  New (FP today, not in frozen list): {sorted(new)}")

    # TP stability -- structural argument + direct check that the same 9 sheet-3 TPs hold.
    print("\nTP stability: contraction (_contract_connectors/_contract_crossings in build.py) "
          "only ADDS edges between existing symbol nodes and removes junction nodes -- it never "
          "calls G.remove_edge on a sym-sym edge, so it cannot structurally cause a TP loss.")
    s3_tp_pairs = []
    r3 = results[3]
    for u, v in r3["s6"].graph.edges():
        gt_u, gt_v = r3["pred_to_gt"].get(u), r3["pred_to_gt"].get(v)
        if gt_u and gt_v and frozenset((gt_u, gt_v)) in r3["gt_edge_set"]:
            s3_tp_pairs.append((u, v))
    print(f"Sheet 3 TP count today: {len(s3_tp_pairs)} (expect 9): {sorted(s3_tp_pairs)}")

    # ------------------------------------------------------------------
    # 4. Does the veto still have a target population?
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DOES THE SR/Sc VETO STILL HAVE A TARGET POPULATION?")
    print("=" * 90)
    total_targets = 0
    for sheet_id in SHEETS:
        targets = results[sheet_id]["veto_targets"]
        total_targets += len(targets)
        print(f"  sheet {sheet_id}: {len(targets)} short-gap pair(s) with sr_conn=True AND "
              f"sc_conn=False{'' if not targets else ': ' + str(targets)}")
    if total_targets == 0:
        print("\n  NONE. Across all three sheets, on today's production output, there is no "
              "short-gap pair (FP or TP) where sr_conn=True and sc_conn=False. The already-wired "
              "run_step4 contraction has no case left for the veto to act on -- it is redundant "
              "given the current crossing/connector detection output.")
    else:
        print(f"\n  {total_targets} candidate(s) found -- veto still has a job.")

    # ------------------------------------------------------------------
    # 5. sym_10 <-> sym_31 investigation
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("INVESTIGATION: sheet 0, sym_10 <-> sym_31 (sr_conn=True, sc_conn=True)")
    print("=" * 90)
    r0 = results[0]
    SR0 = r0["SR"]
    gid_a, gid_b = sym_gid(10), sym_gid(31)
    path, culprits = _diagnose_risk(SR0, gid_a, gid_b)
    print(f"SR shortest path: {' -> '.join(path)}")
    for node in path:
        d = SR0.nodes[node]
        nt = d.get("node_type")
        if nt in ("connector", "crossing"):
            print(f"  {node}: node_type={nt} subtype={d.get('subtype')} "
                  f"cx={d.get('cx'):.0f} cy={d.get('cy'):.0f} "
                  f"pass_through_pairs={d.get('pass_through_pairs')}")

    # Is this pair a GT crossing-separation violation?
    gt_u, gt_v = r0["pred_to_gt"].get(gid_a), r0["pred_to_gt"].get(gid_b)
    print(f"\nGT mapping: {gid_a}->{gt_u}, {gid_b}->{gt_v}")
    is_violation = gt_u is not None and gt_v is not None and frozenset((gt_u, gt_v)) in r0["violation_set"]
    print(f"Is (gt_u, gt_v) a GT crossing-separated pair? {is_violation}")

    junc_nodes_on_path = [n for n in path if n.startswith("junc_")]
    if not junc_nodes_on_path:
        print("\nNo junction node at all lies on the SR path -- sym_10 and sym_31 are connected "
              "by a single continuous (or chained, junction-free) skeleton branch. This is not a "
              "misclassified crossing; it's either a genuine physical connection GT doesn't score, "
              "or ink bleed/bridging between two nearby but unrelated pipes.")
        # Crop around the midpoint of the direct line between the two symbols.
        ax, ay = SR0.nodes[gid_a]["cx"], SR0.nodes[gid_a]["cy"]
        bx, by = SR0.nodes[gid_b]["cx"], SR0.nodes[gid_b]["cy"]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        crop_path = _crop_and_save(r0["png"], mx, my, 0, "sym10_sym31_midpoint", margin=300)
        print(f"Crop (path midpoint) saved -> {crop_path}")
    else:
        for node in junc_nodes_on_path:
            d = SR0.nodes[node]
            crop_path = _crop_and_save(r0["png"], d["cx"], d["cy"], 0, node, margin=300)
            print(f"Crop ({node}, type={d.get('node_type')}) saved -> {crop_path}")


if __name__ == "__main__":
    main()
