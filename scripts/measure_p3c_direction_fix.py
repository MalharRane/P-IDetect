"""Phase 4 -- Fix 1 (P3c direction-aware contraction) measurement.

Compares the OLD all-pairs connector contraction (replicated here exactly as it
existed before the fix, for side-by-side comparison) against the NEW
direction-aware grouping in src/pidetect/graph/build.py::_contract_connectors,
on sheets 0/3/10.

Reports, per sheet:
  - TP/FP/FN/P/R/F1, old vs new
  - Exact TP pair-set diff (lost/gained) -- sheet 3's 9 TPs MUST be unchanged
  - Exact FP pair-set diff (killed/new), each classified into the same
    S4/P3c/P3ab/P3b mechanism buckets as scripts/decompose_p3.py, to confirm
    killed FPs are actually P3c contraction over-bridges and not something else

Usage:
    PYTHONPATH=src python scripts/measure_p3c_direction_fix.py --device cpu
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.build import (
    Step5Result, run_step5, run_step6, sym_gid,
    _oriented_coords, _concat_routes, _constituent_branch_ids,
)
from pidetect.graph.evaluate import load_gt_contracted, match_nodes, edge_metrics

from run_step4_veto_eval import _build_through_s4

CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]

# Frozen doc baseline (docs/phase4_final.md §1, erase_dilation_px=3, pre-Fix-1)
DOC_BASELINE = {
    0:  dict(tp=23, fp=24, fn=8, p=0.489, r=0.742, f1=0.590),
    3:  dict(tp=9,  fp=16, fn=1, p=0.360, r=0.900, f1=0.514),
    10: dict(tp=16, fp=15, fn=6, p=0.516, r=0.727, f1=0.604),
}


# ---------------------------------------------------------------------------
# Exact pre-fix contraction, replicated verbatim for comparison purposes only.
# (This is NOT imported from build.py -- build.py now contains the fix. This
# copy exists solely so "old" and "new" can be measured side by side without
# a git checkout / process round-trip.)
# ---------------------------------------------------------------------------

def _contract_connectors_old_allpairs(G: nx.Graph) -> int:
    connector_gids = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "connector"
    ]
    n_contracted = 0
    for gid in connector_gids:
        if gid not in G:
            continue
        neighbors = list(G.neighbors(gid))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if u == v or G.has_edge(u, v):
                    continue
                edata_u = G.get_edge_data(u, gid)
                edata_v = G.get_edge_data(gid, v)
                coords_u = _oriented_coords(u, gid, edata_u)
                coords_v = _oriented_coords(gid, v, edata_v)
                edge_attrs = dict(contracted=True, via_type="connector")
                if coords_u is not None and coords_v is not None:
                    edge_attrs.update(
                        coords_rc=_concat_routes(coords_u, coords_v),
                        route_u=u, route_v=v,
                        constituent_branch_ids=(
                            _constituent_branch_ids(edata_u) + _constituent_branch_ids(edata_v)
                        ),
                    )
                G.add_edge(u, v, **edge_attrs)
        G.remove_node(gid)
        n_contracted += 1
    return n_contracted


def _run_step6_old(s5: Step5Result):
    from pidetect.graph.build import _contract_crossings, Step6Result
    G = s5.graph.copy()
    n_conn = _contract_connectors_old_allpairs(G)
    n_cross = _contract_crossings(G)
    return Step6Result(graph=G, n_contracted_connectors=n_conn, n_contracted_crossings=n_cross)


# ---------------------------------------------------------------------------
# FP mechanism classification (mirrors scripts/decompose_p3.py)
# ---------------------------------------------------------------------------

def _classify_fp_bucket(gt_u, gt_v, gt_G, violation_set) -> str:
    if frozenset((gt_u, gt_v)) in violation_set:
        return "S4"
    if not nx.has_path(gt_G, gt_u, gt_v):
        return "P3"
    return "OTHER"


def _edge_mechanism(edata: dict) -> str:
    if edata.get("contracted", False):
        return f"contracted:{edata.get('via_type')}"
    if edata.get("short_gap", False):
        return "short_gap"
    return "skeleton_branch"


def _tp_fp_sets(pred_G, gt_G, pred_to_gt, gt_edge_set, violation_set):
    """Return dict pair(sym_a,sym_b, sorted by id) -> (kind, mechanism, bucket) for
    every scoreable pred edge, where kind is 'TP' or 'FP'."""
    out = {}
    for u, v, edata in pred_G.edges(data=True):
        gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
        if gt_u is None or gt_v is None:
            continue
        key = tuple(sorted((u, v)))
        mech = _edge_mechanism(edata)
        if frozenset((gt_u, gt_v)) in gt_edge_set:
            out[key] = ("TP", mech, "")
        else:
            bucket = _classify_fp_bucket(gt_u, gt_v, gt_G, violation_set)
            out[key] = ("FP", mech, bucket)
    return out


def main() -> None:
    import argparse
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
    collinear_tol_deg = cfg["contraction_collinear_tol_deg"]

    print("=" * 100)
    print(f"P3c DIRECTION-AWARE CONTRACTION FIX -- old all-pairs vs new (collinear_tol_deg={collinear_tol_deg})")
    print("=" * 100)

    summary_rows = []
    all_tp_lost = []
    all_new_fp = []

    for sheet_id in SHEETS:
        png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
        )
        s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)

        s6_old = _run_step6_old(s5)
        s6_new = run_step6(s5, collinear_tol_deg)

        gt_G, crossing_infos = load_gt_contracted(gml)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
        violation_set = {
            frozenset((a, b)) for info in crossing_infos for a, b in info["separated_pairs"]
        }

        pred_to_gt_old, gt_to_pred_old = match_nodes(s6_old.graph, gt_G)
        pred_to_gt_new, gt_to_pred_new = match_nodes(s6_new.graph, gt_G)

        em_old = edge_metrics(s6_old.graph, gt_G, pred_to_gt_old, gt_to_pred_old)
        em_new = edge_metrics(s6_new.graph, gt_G, pred_to_gt_new, gt_to_pred_new)

        rows_old = _tp_fp_sets(s6_old.graph, gt_G, pred_to_gt_old, gt_edge_set, violation_set)
        rows_new = _tp_fp_sets(s6_new.graph, gt_G, pred_to_gt_new, gt_edge_set, violation_set)

        tp_old = {k for k, (kind, *_r) in rows_old.items() if kind == "TP"}
        tp_new = {k for k, (kind, *_r) in rows_new.items() if kind == "TP"}
        fp_old = {k: v for k, v in rows_old.items() if v[0] == "FP"}
        fp_new = {k: v for k, v in rows_new.items() if v[0] == "FP"}

        tp_lost = tp_old - tp_new
        tp_gained = tp_new - tp_old
        fp_killed = {k: fp_old[k] for k in fp_old.keys() - fp_new.keys()}
        fp_appeared = {k: fp_new[k] for k in fp_new.keys() - fp_old.keys()}

        print(f"\n--- Sheet {sheet_id} ---")
        db = DOC_BASELINE[sheet_id]
        print(f"  doc baseline (frozen):  TP={db['tp']:<3} FP={db['fp']:<3} FN={db['fn']:<3} "
              f"P={db['p']:.3f} R={db['r']:.3f} F1={db['f1']:.3f}")
        print(f"  old (all-pairs, here):  TP={em_old['tp']:<3} FP={em_old['fp']:<3} FN={em_old['fn']:<3} "
              f"P={em_old['precision']:.3f} R={em_old['recall']:.3f} F1={em_old['f1']:.3f}")
        print(f"  new (direction-aware):  TP={em_new['tp']:<3} FP={em_new['fp']:<3} FN={em_new['fn']:<3} "
              f"P={em_new['precision']:.3f} R={em_new['recall']:.3f} F1={em_new['f1']:.3f}")
        if (em_old['tp'], em_old['fp'], em_old['fn']) != (db['tp'], db['fp'], db['fn']):
            print(f"  NOTE: 'old' replication differs from doc baseline -- drift since freeze, "
                  f"not attributable to this fix.")

        print(f"  TP lost:    {sorted(tp_lost) if tp_lost else 'NONE'}")
        print(f"  TP gained:  {sorted(tp_gained) if tp_gained else 'NONE'}")
        print(f"  FP killed:  {len(fp_killed)}")
        for k, (kind, mech, bucket) in sorted(fp_killed.items()):
            print(f"    {k}  mechanism={mech:<22} bucket={bucket}")
        print(f"  FP appeared: {len(fp_appeared)}")
        for k, (kind, mech, bucket) in sorted(fp_appeared.items()):
            print(f"    {k}  mechanism={mech:<22} bucket={bucket}")

        summary_rows.append((sheet_id, em_old, em_new))
        for k in tp_lost:
            all_tp_lost.append((sheet_id, k))
        for k, v in fp_appeared.items():
            all_new_fp.append((sheet_id, k, v))

    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (per-sheet, old vs new)")
    print("=" * 100)
    print(f"{'sheet':<7}{'old P/R/F1':<26}{'new P/R/F1':<26}{'TP old->new':<14}{'FP old->new'}")
    mean_old_p = mean_old_r = mean_old_f1 = 0.0
    mean_new_p = mean_new_r = mean_new_f1 = 0.0
    for sheet_id, em_old, em_new in summary_rows:
        old_s = f"{em_old['precision']:.3f}/{em_old['recall']:.3f}/{em_old['f1']:.3f}"
        new_s = f"{em_new['precision']:.3f}/{em_new['recall']:.3f}/{em_new['f1']:.3f}"
        print(f"{sheet_id:<7}{old_s:<26}{new_s:<26}"
              f"{em_old['tp']}->{em_new['tp']:<10}{em_old['fp']}->{em_new['fp']}")
        mean_old_p += em_old['precision']; mean_old_r += em_old['recall']; mean_old_f1 += em_old['f1']
        mean_new_p += em_new['precision']; mean_new_r += em_new['recall']; mean_new_f1 += em_new['f1']
    n = len(summary_rows)
    print(f"\nmean (old): P={mean_old_p/n:.3f} R={mean_old_r/n:.3f} F1={mean_old_f1/n:.3f}")
    print(f"mean (new): P={mean_new_p/n:.3f} R={mean_new_r/n:.3f} F1={mean_new_f1/n:.3f}")

    print("\n" + "=" * 100)
    print("TP SAFETY")
    print("=" * 100)
    if all_tp_lost:
        print(f"  !!! {len(all_tp_lost)} TP(s) LOST across all sheets:")
        for sheet_id, k in all_tp_lost:
            print(f"    sheet {sheet_id}: {k}")
    else:
        print("  OK: zero TPs lost on any sheet.")

    if all_new_fp:
        print(f"\n  {len(all_new_fp)} NEW FP(s) appeared (not present in old all-pairs graph):")
        for sheet_id, k, (kind, mech, bucket) in all_new_fp:
            print(f"    sheet {sheet_id}: {k}  mechanism={mech} bucket={bucket}")
    else:
        print("\n  No new FPs appeared on any sheet.")


if __name__ == "__main__":
    main()
