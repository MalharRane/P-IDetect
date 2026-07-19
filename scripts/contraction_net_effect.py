"""Phase 4 Step 4 -- Task 2: quantify contraction's net effect (measurement only).

Compares the PRE-contraction graph (s5: symbol nodes + junction nodes, only
short-gap and direct-skeleton-branch sym-sym edges) against the POST-contraction
graph (s6: today's production graph, junction nodes replaced by all-pairs
connector edges / pass-through-pairs-only crossing edges) on all three sheets.

Since _contract_connectors/_contract_crossings only ADD edges between existing
symbol nodes and never remove a sym-sym edge, s6's edge set is exactly s5's
sym-sym edges plus new edges tagged contracted=True. This script:

  1. Evaluates s5 and s6 against GT independently (TP/FP/FN/P/R/F1).
  2. Isolates the edges that are new in s6 (contracted=True) and are FP, and
     classifies each into S4/P3/OTHER using the same operational definitions as
     scripts/rebaseline_step4.py / scripts/decompose_p3.py.
  3. Isolates the edges that are new in s6 and are TP (contraction's recall gain).
  4. Reports, per sheet: FN fixed vs FP added by contraction, i.e. whether the
     recall gain is worth the precision cost.

Usage:
    PYTHONPATH=src python scripts/contraction_net_effect.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.build import run_step5, run_step6, sym_gid
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, run_step9

from run_step4_veto_eval import _build_through_s4

CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]


def _classify_fp(gt_u, gt_v, gt_G, violation_set) -> str:
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

    print("=" * 100)
    print("TASK 2 -- CONTRACTION NET EFFECT (pre-contraction s5 vs post-contraction s6)")
    print("=" * 100)

    grand = dict(fn_fixed=0, fp_added=0, s4=0, p3=0, other=0)

    for sheet_id in SHEETS:
        png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
        )
        s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        s6 = run_step6(s5)

        s9_pre = run_step9(s5.graph, gml)
        s9_post = run_step9(s6.graph, gml)

        gt_G, crossing_infos = load_gt_contracted(gml)
        pred_to_gt, gt_to_pred = match_nodes(s6.graph, gt_G)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
        violation_set = {
            frozenset((a, b)) for info in crossing_infos for a, b in info["separated_pairs"]
        }

        # Edges present in s6 but not in s5 (pure contraction additions).
        s5_edge_set = {frozenset((u, v)) for u, v in s5.graph.edges()}
        new_edges = [
            (u, v, edata) for u, v, edata in s6.graph.edges(data=True)
            if frozenset((u, v)) not in s5_edge_set
        ]
        # Sanity: every new edge should be tagged contracted=True
        n_untagged = sum(1 for _, _, ed in new_edges if not ed.get("contracted", False))

        new_fp_rows = []
        new_tp_rows = []
        for u, v, edata in new_edges:
            gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
            if gt_u is None or gt_v is None:
                continue  # unscoreable (junction-adjacent node not valve/instrument, etc.)
            if frozenset((gt_u, gt_v)) in gt_edge_set:
                new_tp_rows.append((u, v))
            else:
                bucket = _classify_fp(gt_u, gt_v, gt_G, violation_set)
                new_fp_rows.append((u, v, bucket, edata.get("via_type")))

        print(f"\n--- Sheet {sheet_id} ---")
        print(f"  Pre-contraction  (s5): TP={s9_pre.edge_tp:<3} FP={s9_pre.edge_fp:<3} "
              f"FN={s9_pre.edge_fn:<3} P={s9_pre.edge_precision:.3f} R={s9_pre.edge_recall:.3f} F1={s9_pre.edge_f1:.3f}")
        print(f"  Post-contraction (s6): TP={s9_post.edge_tp:<3} FP={s9_post.edge_fp:<3} "
              f"FN={s9_post.edge_fn:<3} P={s9_post.edge_precision:.3f} R={s9_post.edge_recall:.3f} F1={s9_post.edge_f1:.3f}")
        print(f"  Delta: TP {s9_pre.edge_tp}->{s9_post.edge_tp} ({s9_post.edge_tp - s9_pre.edge_tp:+d})  "
              f"FP {s9_pre.edge_fp}->{s9_post.edge_fp} ({s9_post.edge_fp - s9_pre.edge_fp:+d})  "
              f"FN {s9_pre.edge_fn}->{s9_post.edge_fn} ({s9_post.edge_fn - s9_pre.edge_fn:+d})")
        if n_untagged:
            print(f"  WARNING: {n_untagged} new edge(s) not tagged contracted=True (unexpected)")

        print(f"  New edges added by contraction: {len(new_edges)} total "
              f"({len(new_tp_rows)} scoreable TP, {len(new_fp_rows)} scoreable FP, "
              f"{len(new_edges) - len(new_tp_rows) - len(new_fp_rows)} unscoreable)")

        bucket_counts = {"S4": 0, "P3": 0, "OTHER": 0}
        for u, v, bucket, via in new_fp_rows:
            bucket_counts[bucket] += 1
        print(f"  New-FP bucket split: S4={bucket_counts['S4']} P3={bucket_counts['P3']} OTHER={bucket_counts['OTHER']}")
        if new_fp_rows:
            print(f"  New FP pairs: {[(u, v, b) for u, v, b, _ in new_fp_rows]}")
        if new_tp_rows:
            print(f"  New TP pairs (contraction fixed these FNs): {new_tp_rows}")

        grand["fn_fixed"] += len(new_tp_rows)
        grand["fp_added"] += len(new_fp_rows)
        grand["s4"] += bucket_counts["S4"]
        grand["p3"] += bucket_counts["P3"]
        grand["other"] += bucket_counts["OTHER"]

    print("\n" + "=" * 100)
    print("TOTALS ACROSS 3 SHEETS")
    print("=" * 100)
    print(f"  FNs fixed by contraction (new scoreable TPs): {grand['fn_fixed']}")
    print(f"  FPs added by contraction (new scoreable FPs): {grand['fp_added']}")
    print(f"    of which S4 (still a crossing-violation even after contraction): {grand['s4']}")
    print(f"    of which P3 (no GT path at all -- P3c over-bridge):              {grand['p3']}")
    print(f"    of which OTHER:                                                  {grand['other']}")
    ratio = grand["fn_fixed"] / grand["fp_added"] if grand["fp_added"] else float("inf")
    print(f"  Ratio TP-gained : FP-added = {grand['fn_fixed']}:{grand['fp_added']} ({ratio:.2f})")


if __name__ == "__main__":
    main()
