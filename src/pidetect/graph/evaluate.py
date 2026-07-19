"""Phase 4 step 9: evaluate predicted contracted graph against PID2Graph GT.

GT contraction order (matches PID2Graph annotation semantics):
  1. Remove background nodes (title-block bounding boxes, not connectivity)
  2. Contract connector nodes (standard: connect all neighbour pairs, remove)
  3. Contract general nodes (same semantics as connector in GT)
  4. Contract crossing nodes with pass-through semantics only:
       for each crossing, find its pass-through neighbour pairs by collinearity,
       add only those edges, then remove the crossing node.

Public API:
    load_gt_contracted(gml_path) -> (G, crossing_infos)
    match_nodes(pred_G, gt_G) -> (pred_to_gt, gt_to_pred)
    edge_metrics(pred_G, gt_G, pred_to_gt, gt_to_pred) -> dict
    crossing_nonedge_rate(pred_G, crossing_infos, gt_to_pred) -> dict
    run_step9(pred_G, gml_path) -> Step9Result
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _gt_centroid(attrs: dict) -> tuple[float, float]:
    x1 = float(attrs.get("xmin", 0)); x2 = float(attrs.get("xmax", 0))
    y1 = float(attrs.get("ymin", 0)); y2 = float(attrs.get("ymax", 0))
    return (x1 + x2) / 2, (y1 + y2) / 2


def _gt_bbox_wh(attrs: dict) -> tuple[float, float]:
    return (
        abs(float(attrs.get("xmax", 0)) - float(attrs.get("xmin", 0))),
        abs(float(attrs.get("ymax", 0)) - float(attrs.get("ymin", 0))),
    )


def _anti_parallel_err(
    cx: float, cy: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Angle deviation (degrees) from 180° for the A–crossing–B configuration. Lower = more collinear."""
    ang_a = math.atan2(ay - cy, ax - cx)
    ang_b = math.atan2(by - cy, bx - cx)
    diff = math.degrees(ang_b - ang_a) % 360
    return abs(diff - 180.0)


def _best_passthrough_pairs(
    neighbors: list[str],
    cx: float,
    cy: float,
    G: nx.Graph,
) -> list[tuple[str, str]]:
    """Geometric pass-through pair selection for a crossing at (cx, cy).

    For n == 2: trivial 1 pair.
    For n == 3: find the single best anti-parallel pair.
    For n >= 4: try all 3 pairings of the first 4 neighbours; pick min total anti-parallel error.
    """
    n = len(neighbors)
    if n <= 1:
        return []
    if n == 2:
        return [(neighbors[0], neighbors[1])]
    if n == 3:
        best_err = float("inf")
        best_pair: tuple[str, str] = (neighbors[0], neighbors[1])
        for a, b in itertools.combinations(neighbors, 2):
            ax, ay = _gt_centroid(G.nodes[a])
            bx, by = _gt_centroid(G.nodes[b])
            err = _anti_parallel_err(cx, cy, ax, ay, bx, by)
            if err < best_err:
                best_err = err
                best_pair = (a, b)
        return [best_pair]
    # n == 4 (or more): enumerate all 3 pairings of first 4 neighbours
    a, b, c, d = neighbors[:4]
    candidates = [
        [(a, b), (c, d)],
        [(a, c), (b, d)],
        [(a, d), (b, c)],
    ]
    best_total = float("inf")
    best: list[tuple[str, str]] = candidates[0]
    for pairing in candidates:
        total_err = sum(
            _anti_parallel_err(
                cx, cy,
                *_gt_centroid(G.nodes[u]),
                *_gt_centroid(G.nodes[v]),
            )
            for u, v in pairing
        )
        if total_err < best_total:
            best_total = total_err
            best = pairing
    return [tuple(p) for p in best]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# GT loading and contraction
# ---------------------------------------------------------------------------

_SCORE_LABELS: frozenset[str] = frozenset({"valve", "instrumentation"})


def _contract_label(G: nx.Graph, label: str) -> None:
    """Standard contraction: connect all neighbour pairs, then remove node. In-place."""
    nodes = [n for n, d in G.nodes(data=True) if d.get("label") == label]
    for nid in nodes:
        if nid not in G:
            continue
        nbrs = list(G.neighbors(nid))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                u, v = nbrs[i], nbrs[j]
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        G.remove_node(nid)


def _contract_crossings_gt(G: nx.Graph) -> list[dict]:
    """Contract GT crossing nodes with pass-through semantics.

    Returns a list of crossing_info dicts:
        {"id", "cx", "cy", "pass_through_pairs": [(A,B),...], "separated_pairs": [(X,Y),...]}
    """
    crossing_ids = [n for n, d in G.nodes(data=True) if d.get("label") == "crossing"]
    infos: list[dict] = []

    for cid in crossing_ids:
        if cid not in G:
            continue
        cx, cy = _gt_centroid(G.nodes[cid])
        nbrs = list(G.neighbors(cid))
        pass_pairs = _best_passthrough_pairs(nbrs, cx, cy, G)
        pass_set = {frozenset(p) for p in pass_pairs}

        sep_pairs = [
            (a, b)
            for a, b in itertools.combinations(nbrs, 2)
            if frozenset((a, b)) not in pass_set
        ]

        for a, b in pass_pairs:
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b)
        G.remove_node(cid)

        infos.append({
            "id": cid,
            "cx": cx,
            "cy": cy,
            "pass_through_pairs": list(pass_pairs),
            "separated_pairs": sep_pairs,
        })

    return infos


def load_gt_contracted(gml_path: Path) -> tuple[nx.Graph, list[dict]]:
    """Load GT GraphML and return the contracted symbol graph with crossing metadata.

    Contraction order: remove background → contract connector → contract general
    → contract crossing (with pass-through semantics only).

    Returns:
        G:              NetworkX graph; only symbol nodes remain as vertices.
        crossing_infos: list of crossing dicts; each has "separated_pairs"
                        — GT neighbour pairs that must NOT be connected.
    """
    G = nx.read_graphml(str(gml_path))

    bg = [n for n, d in G.nodes(data=True) if d.get("label") == "background"]
    G.remove_nodes_from(bg)

    _contract_label(G, "connector")
    _contract_label(G, "general")
    crossing_infos = _contract_crossings_gt(G)
    return G, crossing_infos


# ---------------------------------------------------------------------------
# Node matching (CtrMt@50%)
# ---------------------------------------------------------------------------

# Our pred node_type → GT label for matching
_PRED_TO_GT_LABEL: dict[str, str] = {
    "valve":      "valve",
    "instrument": "instrumentation",
}


def match_nodes(
    pred_G: nx.Graph,
    gt_G: nx.Graph,
) -> tuple[dict[str, str], dict[str, str]]:
    """Greedy CtrMt@50% bipartite matching for valve and instrumentation nodes.

    Pred nodes sorted by confidence (desc); greedily claim the closest eligible GT node
    within 0.5 * sqrt(gt_w * gt_h).  One-to-one; same label only.

    Returns:
        pred_to_gt: {pred_gid -> gt_gid}
        gt_to_pred: {gt_gid -> pred_gid}
    """
    # Index GT nodes by label
    gt_by_label: dict[str, list[str]] = {}
    for gid, attrs in gt_G.nodes(data=True):
        lbl = attrs.get("label", "")
        if lbl in _SCORE_LABELS:
            gt_by_label.setdefault(lbl, []).append(gid)

    # Sort pred nodes by confidence descending
    pred_scored = sorted(
        [
            (gid, attrs)
            for gid, attrs in pred_G.nodes(data=True)
            if attrs.get("node_type") in _PRED_TO_GT_LABEL
        ],
        key=lambda x: float(x[1].get("conf", 0.0)),
        reverse=True,
    )

    pred_to_gt: dict[str, str] = {}
    gt_to_pred: dict[str, str] = {}
    used_gt: set[str] = set()

    for pred_gid, pred_attrs in pred_scored:
        gt_label = _PRED_TO_GT_LABEL[pred_attrs["node_type"]]
        candidates = gt_by_label.get(gt_label, [])
        if not candidates:
            continue

        px = (float(pred_attrs.get("x1", 0)) + float(pred_attrs.get("x2", 0))) / 2
        py = (float(pred_attrs.get("y1", 0)) + float(pred_attrs.get("y2", 0))) / 2

        best_dist = float("inf")
        best_gt: str | None = None
        for gt_gid in candidates:
            if gt_gid in used_gt:
                continue
            gt_attrs = gt_G.nodes[gt_gid]
            gw, gh = _gt_bbox_wh(gt_attrs)
            threshold = 0.5 * math.sqrt(gw * gh)
            gcx, gcy = _gt_centroid(gt_attrs)
            dist = math.hypot(px - gcx, py - gcy)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best_gt = gt_gid

        if best_gt is not None:
            pred_to_gt[pred_gid] = best_gt
            gt_to_pred[best_gt] = pred_gid
            used_gt.add(best_gt)

    return pred_to_gt, gt_to_pred


# ---------------------------------------------------------------------------
# Edge metrics
# ---------------------------------------------------------------------------

def edge_metrics(
    pred_G: nx.Graph,
    gt_G: nx.Graph,
    pred_to_gt: dict[str, str],
    gt_to_pred: dict[str, str],
) -> dict[str, int | float]:
    """TP/FP/FN and P/R/F1 on edges where both endpoints are matched.

    An edge is scoreable only when both its endpoints appear in the matching.
    Edges touching unmatched nodes (unknown_fitting, off_page, vessel) are skipped.
    """
    gt_edge_set: set[frozenset] = {frozenset((u, v)) for u, v in gt_G.edges()}

    # Track which GT edges are covered by a pred edge
    covered_gt: set[frozenset] = set()
    tp = fp = 0

    for u, v in pred_G.edges():
        gt_u = pred_to_gt.get(u)
        gt_v = pred_to_gt.get(v)
        if gt_u is None or gt_v is None:
            continue
        key = frozenset((gt_u, gt_v))
        if key in gt_edge_set:
            tp += 1
            covered_gt.add(key)
        else:
            fp += 1

    # FN = GT edges with both endpoints matched but no covering pred edge
    fn = 0
    for u, v in gt_G.edges():
        pu = gt_to_pred.get(u)
        pv = gt_to_pred.get(v)
        if pu is not None and pv is not None:
            if frozenset((u, v)) not in covered_gt:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Crossing non-edge rate
# ---------------------------------------------------------------------------

def crossing_nonedge_rate(
    pred_G: nx.Graph,
    crossing_infos: list[dict],
    gt_to_pred: dict[str, str],
) -> dict[str, int | float]:
    """Fraction of GT crossing separated-pairs that appear as a predicted edge.

    A "violation" is a crossing-separated pair (A, B) — which should NOT be connected —
    that appears as an edge in the predicted graph after mapping A and B to pred nodes.
    Only pairs where both GT nodes have a pred match are counted.
    """
    total = violations = 0

    for info in crossing_infos:
        for gt_a, gt_b in info["separated_pairs"]:
            pred_a = gt_to_pred.get(gt_a)
            pred_b = gt_to_pred.get(gt_b)
            if pred_a is None or pred_b is None:
                continue
            total += 1
            if pred_G.has_edge(pred_a, pred_b):
                violations += 1

    rate = violations / total if total > 0 else 0.0
    return {"total": total, "violations": violations, "rate": rate}


# ---------------------------------------------------------------------------
# Step 9 result
# ---------------------------------------------------------------------------

@dataclass
class Step9Result:
    # GT baseline
    n_gt_scoreable: int       # valve + instrumentation nodes in contracted GT
    n_gt_edges_scoreable: int # GT edges where both endpoints have a pred match
    # Matching
    n_pred_matched: int
    n_gt_matched: int
    # Edge metrics (on matched pairs only)
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_precision: float
    edge_recall: float
    edge_f1: float
    # Crossing
    n_crossing_sep_pairs: int
    n_crossing_violations: int
    crossing_nonedge_rate: float  # violation rate; lower is better

    @property
    def match_recall(self) -> float:
        return self.n_gt_matched / self.n_gt_scoreable if self.n_gt_scoreable > 0 else 0.0

    def summary_lines(self) -> list[str]:
        return [
            f"  Node match: {self.n_pred_matched}/{self.n_gt_scoreable} GT scoreable"
            f"  (recall {self.match_recall:.1%})",
            f"  Edges (matched pairs): TP={self.edge_tp} FP={self.edge_fp}"
            f" FN={self.edge_fn}  [{self.n_gt_edges_scoreable} GT edges scoreable]",
            f"    Precision={self.edge_precision:.3f}  Recall={self.edge_recall:.3f}"
            f"  F1={self.edge_f1:.3f}",
            f"  Crossing sep-pairs: {self.n_crossing_sep_pairs}"
            f"  violations={self.n_crossing_violations}"
            f"  non-edge rate={1 - self.crossing_nonedge_rate:.1%}"
            f"  (cross-connect rate={self.crossing_nonedge_rate:.1%})",
        ]


def run_step9(pred_G: nx.Graph, gml_path: Path) -> Step9Result:
    """Evaluate the contracted predicted graph against the GT GraphML.

    Args:
        pred_G:   contracted NetworkX graph from Step 6.
        gml_path: path to the raw GT .graphml file for this sheet.

    Returns:
        Step9Result with edge P/R/F1 and crossing non-edge rate.
    """
    gt_G, crossing_infos = load_gt_contracted(gml_path)

    n_gt_scoreable = sum(
        1 for _, d in gt_G.nodes(data=True) if d.get("label") in _SCORE_LABELS
    )

    pred_to_gt, gt_to_pred = match_nodes(pred_G, gt_G)

    em = edge_metrics(pred_G, gt_G, pred_to_gt, gt_to_pred)

    n_gt_edges_scoreable = sum(
        1 for u, v in gt_G.edges()
        if gt_to_pred.get(u) is not None and gt_to_pred.get(v) is not None
    )

    cr = crossing_nonedge_rate(pred_G, crossing_infos, gt_to_pred)

    return Step9Result(
        n_gt_scoreable=n_gt_scoreable,
        n_gt_edges_scoreable=n_gt_edges_scoreable,
        n_pred_matched=len(pred_to_gt),
        n_gt_matched=len(gt_to_pred),
        edge_tp=em["tp"],
        edge_fp=em["fp"],
        edge_fn=em["fn"],
        edge_precision=em["precision"],
        edge_recall=em["recall"],
        edge_f1=em["f1"],
        n_crossing_sep_pairs=cr["total"],
        n_crossing_violations=cr["violations"],
        crossing_nonedge_rate=cr["rate"],
    )
