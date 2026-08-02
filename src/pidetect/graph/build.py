"""Phase 4 steps 5–7: graph construction, contraction, and direction tagging.

Step 5 — build_graph():
  Nodes: sym_N (symbol/off-page, N = SymbolNode.node_id) +
         junc_J (connector/crossing, J = JunctionNode.junc_id)
  Edges: one per skeleton branch whose both endpoints map to a node.
  Attributes: branch_id, length_px, branch_type, coords_rc (polyline)

Step 6 — contraction:
  Connector nodes: remove, connect all neighbour pairs (standard topology contraction).
  Crossing nodes: remove, connect only pass_through_pairs (preserves non-connection).

Step 7 — direction tagging:
  For each flow_arrow: find closest edge in contracted graph, infer tip side from
  half-bbox dark-pixel count, tag edge with flow_direction = "src_gid -> dst_gid".

Step 4b (crossing-separation veto) — see docs/phase4_step4_scope.md §3:
  Short-gap pairs bypass crossing contraction (they're inserted as direct sym-to-sym
  edges with no intermediate junction node — contraction never sees them). The veto
  suppresses a short-gap pair (A, B) iff the raw skeleton graph SR connects A to B
  but the contracted graph Sc (SR with connectors + crossings contracted) does not —
  i.e. the only skeleton path between A and B is broken by a crossing's pass-through
  semantics.

Public API:
    run_step5(all_symbol_nodes, off_page_nodes, s3, s4) -> Step5Result
    run_step6(s5)                                        -> Step6Result
    run_step7(G, flow_arrows, img_gray, max_dist_px)     -> Step7Result
    build_sr_sc(all_symbol_nodes, off_page_nodes, s3, s4)                -> (SR, Sc)
    veto_short_gap_by_crossing_separation(short_gap_pairs, SR, Sc)       -> (kept, suppressed)
"""
from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from pidetect.graph.erase import SymbolNode
from pidetect.graph.lines import Branch, Step3Result
from pidetect.graph.junction import Step4Result


# ---------------------------------------------------------------------------
# Node ID helpers (avoid collisions between sym and junc integer spaces)
# ---------------------------------------------------------------------------

def sym_gid(node_id: int) -> str:
    return f"sym_{node_id}"


def junc_gid(junc_id: int) -> str:
    return f"junc_{junc_id}"


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# ---------------------------------------------------------------------------
# Step 5a: endpoint → graph-node mapping
# ---------------------------------------------------------------------------

def _build_endpoint_map(
    s3: Step3Result,
    s4: Step4Result,
) -> dict[tuple[int, str], Optional[str]]:
    """Map (branch_id, which_end) -> graph node ID string.

    "sym_N"   — endpoint bound to symbol node N (step 3)
    "junc_J"  — endpoint at junction J (step 4)
    None      — dangling: unbound and not at any detected junction
    """
    # Seed from step-3 symbol bindings
    ep_map: dict[tuple[int, str], Optional[str]] = {}
    for ep in s3.endpoints:
        key = (ep.branch_id, ep.which_end)
        ep_map[key] = sym_gid(ep.bound_node_id) if ep.bound_node_id is not None else None

    # Build branch lookup for proximity check
    branch_by_id: dict[int, Branch] = {b.branch_id: b for b in s3.branches}

    # Augment with junction nodes — only fills entries still None
    for jn in s4.all_junctions:
        jgid = junc_gid(jn.junc_id)
        for bid in jn.branch_ids:
            b = branch_by_id.get(bid)
            if b is None:
                continue
            src_key = (bid, "src")
            dst_key = (bid, "dst")
            # The closer branch endpoint is the one touching this junction
            src_d = _dist(b.src_xy[0], b.src_xy[1], jn.cx, jn.cy)
            dst_d = _dist(b.dst_xy[0], b.dst_xy[1], jn.cx, jn.cy)
            if src_d <= dst_d:
                if ep_map.get(src_key) is None:
                    ep_map[src_key] = jgid
            else:
                if ep_map.get(dst_key) is None:
                    ep_map[dst_key] = jgid

    return ep_map


# ---------------------------------------------------------------------------
# Step 5b: graph construction
# ---------------------------------------------------------------------------

def build_graph(
    all_symbol_nodes: list[SymbolNode],
    off_page_nodes: list[SymbolNode],
    s3: Step3Result,
    s4: Step4Result,
) -> tuple[nx.Graph, int]:
    """Build the raw connectivity graph.

    Returns (G, n_dangling) where n_dangling is the count of branches skipped
    because at least one endpoint could not be mapped to any node.
    """
    G = nx.Graph()

    # --- symbol nodes (YOLO detections + GT vessel bodies) ---
    for node in all_symbol_nodes:
        if not node.is_graph_node:
            continue  # flow_arrow / tag_rect: excluded from graph
        attrs = dict(
            node_type=node.node_type,
            cls_name=node.cls_name,
            cx=float(node.cx),
            cy=float(node.cy),
            x1=float(node.x1), y1=float(node.y1),
            x2=float(node.x2), y2=float(node.y2),
            conf=float(node.conf),
            n_ports=len(node.ports),
        )
        if node.node_type == "instrument":
            # Phase 3 (docs/phase3_design.md) -- present only when pidetect.text.ocr
            # actually ran; tag_parse_status defaults to "not_run" otherwise (never
            # fabricated as "ok").
            attrs.update(
                tag_raw_text=node.tag_raw_text,
                tag_function=node.tag_function,
                tag_loop_number=node.tag_loop_number,
                tag_confidence=float(node.tag_confidence),
                tag_parse_status=node.tag_parse_status,
            )
        G.add_node(sym_gid(node.node_id), **attrs)

    # --- off-page nodes (synthetic border terminals from step 3) ---
    for node in off_page_nodes:
        G.add_node(sym_gid(node.node_id),
                   node_type="off_page",
                   cls_name="off_page",
                   cx=float(node.cx),
                   cy=float(node.cy),
                   x1=float(node.x1), y1=float(node.y1),
                   x2=float(node.x2), y2=float(node.y2),
                   conf=1.0,
                   n_ports=len(node.ports))

    # --- junction nodes (connectors and crossings from step 4) ---
    for jn in s4.all_junctions:
        G.add_node(junc_gid(jn.junc_id),
                   node_type=jn.junc_type,        # "connector" | "crossing"
                   subtype=jn.subtype,
                   cls_name=jn.junc_type,
                   cx=float(jn.cx),
                   cy=float(jn.cy),
                   pass_through_pairs=jn.pass_through_pairs)

    # --- build endpoint -> node map ---
    ep_map = _build_endpoint_map(s3, s4)

    # --- edges: one per branch with both endpoints mapped ---
    n_dangling = 0
    for branch in s3.branches:
        src_gid = ep_map.get((branch.branch_id, "src"))
        dst_gid = ep_map.get((branch.branch_id, "dst"))
        if src_gid is None or dst_gid is None:
            n_dangling += 1
            continue
        if src_gid == dst_gid:
            continue  # degenerate self-loop
        if not G.has_edge(src_gid, dst_gid):
            G.add_edge(src_gid, dst_gid,
                       branch_id=branch.branch_id,
                       length_px=float(branch.length_px),
                       branch_type=int(branch.branch_type),
                       coords_rc=branch.coords_rc,  # (N,2) polyline for geometry
                       route_u=src_gid,             # orientation: coords_rc runs route_u -> route_v
                       route_v=dst_gid)

    # --- short-gap edges: pipe fully erased by overlapping dilated bboxes ---
    # These pairs were confirmed connected in the pre-erasure binary image.
    node_id_to_gid = {node.node_id: sym_gid(node.node_id)
                      for node in all_symbol_nodes + off_page_nodes
                      if node.is_graph_node}
    for id_a, id_b in s3.short_gap_pairs:
        gid_a = node_id_to_gid.get(id_a)
        gid_b = node_id_to_gid.get(id_b)
        if gid_a is None or gid_b is None:
            continue
        if gid_a == gid_b:
            continue
        if not G.has_edge(gid_a, gid_b):
            G.add_edge(gid_a, gid_b, branch_type=-1, short_gap=True)

    return G, n_dangling


# ---------------------------------------------------------------------------
# Step 5 result + entry point
# ---------------------------------------------------------------------------

@dataclass
class Step5Result:
    graph: nx.Graph
    n_dangling: int   # branches skipped (>=1 unbound endpoint)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def node_type_counts(self) -> dict[str, int]:
        return dict(Counter(d["node_type"] for _, d in self.graph.nodes(data=True)))

    def components_summary(self) -> dict:
        ccs = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        sizes = [len(cc) for cc in ccs]
        return {
            "n_components": len(sizes),
            "largest":      sizes[0] if sizes else 0,
            "isolated":     sum(1 for s in sizes if s == 1),
            "sizes_top5":   sizes[:5],
        }


def run_step5(
    all_symbol_nodes: list[SymbolNode],
    off_page_nodes: list[SymbolNode],
    s3: Step3Result,
    s4: Step4Result,
) -> Step5Result:
    """Build the raw pre-contraction connectivity graph (step 5)."""
    G, n_dangling = build_graph(all_symbol_nodes, off_page_nodes, s3, s4)
    return Step5Result(graph=G, n_dangling=n_dangling)


# ---------------------------------------------------------------------------
# Route tracing helpers — reconstruct the real traced polyline through a
# contracted chain of junction nodes, instead of discarding it (docs/phase4_final.md
# fix: contracted edges were falling back to a straight centroid-to-centroid chord).
# ---------------------------------------------------------------------------

def _oriented_coords(node_from: str, node_to: str, edata: dict) -> Optional[list]:
    """Return edata's coords_rc as a plain [[row, col], ...] list, ordered node_from -> node_to.

    None when the edge carries no reconstructable path (short-gap edges, or a
    prior contraction step that itself couldn't trace a route).
    """
    coords = edata.get("coords_rc")
    if coords is None or len(coords) == 0:
        return None
    route_u, route_v = edata.get("route_u"), edata.get("route_v")
    pts = [[float(r), float(c)] for r, c in coords]
    if route_u == node_from and route_v == node_to:
        return pts
    if route_u == node_to and route_v == node_from:
        return pts[::-1]
    return None  # orientation unknown -- shouldn't happen for edges we constructed


def _concat_routes(coords_a: list, coords_b: list) -> list:
    """Concatenate two node_from->node_to polylines sharing their joining point."""
    if coords_a and coords_b and coords_a[-1] == coords_b[0]:
        return coords_a + coords_b[1:]
    return coords_a + coords_b


def _constituent_branch_ids(edata: dict) -> list:
    """Ordered list of original skeleton branch IDs folded into this edge so far."""
    if "constituent_branch_ids" in edata:
        return list(edata["constituent_branch_ids"])
    if "branch_id" in edata:
        return [edata["branch_id"]]
    return []


# ---------------------------------------------------------------------------
# Step 6a: connector contraction (standard — connect all neighbour pairs)
# ---------------------------------------------------------------------------

def _contract_connectors(G: nx.Graph) -> int:
    """Contract all connector nodes in G in-place.

    For each connector C: add edges between every pair of C's neighbours,
    then remove C.  This is the standard topology-node contraction for
    pipe bends and T-junctions. When both legs (neighbour->C) carry a traced
    polyline, the two are concatenated onto the new edge so the contracted
    edge still renders the real pipe route instead of a straight chord.

    Returns the number of connector nodes contracted.
    """
    connector_gids = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "connector"
    ]
    n_contracted = 0
    for gid in connector_gids:
        if gid not in G:
            continue
        neighbors = list(G.neighbors(gid))  # materialise before removal
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
                        route_u=u,
                        route_v=v,
                        constituent_branch_ids=(
                            _constituent_branch_ids(edata_u) + _constituent_branch_ids(edata_v)
                        ),
                    )
                G.add_edge(u, v, **edge_attrs)
        G.remove_node(gid)
        n_contracted += 1
    return n_contracted


# ---------------------------------------------------------------------------
# Step 6b: crossing contraction (pass-through pairs only — no cross-edges)
# ---------------------------------------------------------------------------

def _contract_crossings(G: nx.Graph) -> int:
    """Contract all crossing nodes in G in-place using pass-through semantics.

    For crossing C with pass_through_pairs = [(bid_A, bid_B), (bid_X, bid_Y)]:
      - Find the neighbour connected to C via branch bid_A  →  nbr_A
      - Find the neighbour connected to C via branch bid_B  →  nbr_B
      - Add edge (nbr_A, nbr_B) — the underpass pipe is continuous
      - Same for (bid_X, bid_Y) — the overpass pipe is continuous
      - Do NOT add (nbr_A, nbr_X) etc. — the pipes don't join at a crossing

    Returns the number of crossing nodes contracted.
    """
    crossing_items = [
        (n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "crossing"
    ]
    n_contracted = 0
    for gid, attrs in crossing_items:
        if gid not in G:
            continue
        # Map branch_id -> (neighbour, edge attrs) for every edge incident to this crossing
        branch_to_edge: dict[int, tuple[str, dict]] = {}
        for u, v, edata in G.edges(gid, data=True):
            bid = edata.get("branch_id")
            if bid is not None:
                nbr = v if u == gid else u
                branch_to_edge[bid] = (nbr, edata)

        for (bid_A, bid_B) in attrs.get("pass_through_pairs", []):
            if bid_A == bid_B:
                continue  # degenerate single-overpass entry — skip
            entry_A = branch_to_edge.get(bid_A)
            entry_B = branch_to_edge.get(bid_B)
            if entry_A is None or entry_B is None:
                continue
            nbr_A, edata_A = entry_A
            nbr_B, edata_B = entry_B
            if nbr_A == nbr_B or G.has_edge(nbr_A, nbr_B):
                continue
            coords_A = _oriented_coords(nbr_A, gid, edata_A)
            coords_B = _oriented_coords(gid, nbr_B, edata_B)
            edge_attrs = dict(contracted=True, via_type="crossing")
            if coords_A is not None and coords_B is not None:
                edge_attrs.update(
                    coords_rc=_concat_routes(coords_A, coords_B),
                    route_u=nbr_A,
                    route_v=nbr_B,
                    constituent_branch_ids=(
                        _constituent_branch_ids(edata_A) + _constituent_branch_ids(edata_B)
                    ),
                )
            G.add_edge(nbr_A, nbr_B, **edge_attrs)

        G.remove_node(gid)
        n_contracted += 1
    return n_contracted


# ---------------------------------------------------------------------------
# Step 6 result + entry point
# ---------------------------------------------------------------------------

@dataclass
class Step6Result:
    graph: nx.Graph              # contracted symbol-to-symbol graph
    n_contracted_connectors: int
    n_contracted_crossings: int

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def node_type_counts(self) -> dict[str, int]:
        return dict(Counter(d["node_type"] for _, d in self.graph.nodes(data=True)))

    def components_summary(self) -> dict:
        ccs = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        sizes = [len(cc) for cc in ccs]
        return {
            "n_components": len(sizes),
            "largest":      sizes[0] if sizes else 0,
            "isolated":     sum(1 for s in sizes if s == 1),
            "sizes_top5":   sizes[:5],
        }


def run_step6(s5: Step5Result) -> Step6Result:
    """Contract connector and crossing nodes to produce the symbol-to-symbol graph."""
    G = s5.graph.copy()
    n_conn  = _contract_connectors(G)
    n_cross = _contract_crossings(G)
    return Step6Result(
        graph=G,
        n_contracted_connectors=n_conn,
        n_contracted_crossings=n_cross,
    )


# ---------------------------------------------------------------------------
# Step 4b: short-gap crossing-separation veto (docs/phase4_step4_scope.md §3)
# ---------------------------------------------------------------------------

def build_sr_sc(
    all_symbol_nodes: list[SymbolNode],
    off_page_nodes: list[SymbolNode],
    s3: Step3Result,
    s4: Step4Result,
) -> tuple[nx.Graph, nx.Graph]:
    """Build the two auxiliary graphs used by the crossing-separation veto.

    SR — raw skeleton graph, built with short_gap_pairs=[] (skeleton-traced
         branch edges + junction nodes only; no short-gap edges).
    Sc — SR with connector and crossing nodes contracted (pass-through semantics
         for crossings), i.e. what step 6 would produce from SR alone.
    """
    s3_no_sg = dataclasses.replace(s3, short_gap_pairs=[])
    SR, _n_dangling = build_graph(all_symbol_nodes, off_page_nodes, s3_no_sg, s4)
    Sc = SR.copy()
    _contract_connectors(Sc)
    _contract_crossings(Sc)
    return SR, Sc


def veto_short_gap_by_crossing_separation(
    short_gap_pairs: list[tuple[int, int]],
    SR: nx.Graph,
    Sc: nx.Graph,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Suppress short-gap pairs whose only skeleton path is crossing-separated.

    Pair (A, B) is suppressed iff has_path(SR, A, B) and NOT has_path(Sc, A, B):
    A and B are reachable through the raw skeleton, but that path is broken once
    crossings are contracted with pass-through semantics — the pipe network
    routes A toward B via a different pipe that only appears connected because
    a crossing (not a real junction) sits between them.

    Returns (kept_pairs, suppressed_pairs).
    """
    kept: list[tuple[int, int]] = []
    suppressed: list[tuple[int, int]] = []
    for id_a, id_b in short_gap_pairs:
        gid_a, gid_b = sym_gid(id_a), sym_gid(id_b)
        sr_conn = gid_a in SR and gid_b in SR and nx.has_path(SR, gid_a, gid_b)
        sc_conn = gid_a in Sc and gid_b in Sc and nx.has_path(Sc, gid_a, gid_b)
        if sr_conn and not sc_conn:
            suppressed.append((id_a, id_b))
        else:
            kept.append((id_a, id_b))
    return kept, suppressed


# ---------------------------------------------------------------------------
# Step 7: direction tagging from flow-arrow predictions
# ---------------------------------------------------------------------------

def _segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Perpendicular distance from point P to line segment AB, clamped at endpoints."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)))
    return math.hypot(px - (ax + t*dx), py - (ay + t*dy))


def _find_adjacent_edge(
    acx: float,
    acy: float,
    G: nx.Graph,
    max_dist_px: float,
) -> Optional[tuple[str, str]]:
    """Find the contracted-graph edge whose sym-to-sym segment is closest to (acx, acy).

    Uses perpendicular distance from the arrow centre to the straight-line segment
    connecting the two symbol centroids.  Returns None if the closest edge is
    further than max_dist_px.
    """
    best: Optional[tuple[str, str]] = None
    best_d = float("inf")
    for u, v in G.edges():
        ua, va = G.nodes[u], G.nodes[v]
        d = _segment_dist(
            acx, acy,
            ua.get("cx", 0.0), ua.get("cy", 0.0),
            va.get("cx", 0.0), va.get("cy", 0.0),
        )
        if d < best_d:
            best_d = d
            best = (u, v)
    return best if best_d <= max_dist_px else None


def _arrow_tip_xy(
    arrow: SymbolNode,
    img_gray: np.ndarray,
) -> tuple[float, float]:
    """Return (tip_x, tip_y) — the inferred arrowhead-tip coordinate.

    Splits the arrow bbox along its long axis into two halves.
    The half with more dark (< 128) pixels contains the filled arrowhead triangle.
    For horizontal arrows: tip at right (x2) if right half is darker, else left (x1).
    For vertical arrows: tip at bottom (y2) if bottom half is darker, else top (y1).
    """
    x1, y1, x2, y2 = int(arrow.x1), int(arrow.y1), int(arrow.x2), int(arrow.y2)
    H, W = img_gray.shape[:2]
    roi = img_gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if roi.size == 0:
        return arrow.cx, arrow.cy

    dark = roi < 128
    rh, rw = dark.shape
    mid_cx = (x1 + x2) / 2.0
    mid_cy = (y1 + y2) / 2.0

    if rw >= rh:   # horizontal arrow — compare left vs right half
        half = rw // 2
        if dark[:, half:].sum() > dark[:, :half].sum():
            return float(x2), mid_cy   # arrowhead at right
        else:
            return float(x1), mid_cy   # arrowhead at left
    else:          # vertical arrow — compare top vs bottom half
        half = rh // 2
        if dark[half:, :].sum() > dark[:half, :].sum():
            return mid_cx, float(y2)   # arrowhead at bottom
        else:
            return mid_cx, float(y1)   # arrowhead at top


@dataclass
class Step7Result:
    n_arrows_total:   int   # flow_arrow SymbolNodes received
    n_arrows_assigned: int  # matched to an edge within max_dist
    n_arrows_directed: int  # direction successfully determined (image provided)
    n_edges_directed:  int  # distinct edges that received a flow_direction tag


def run_step7(
    G: nx.Graph,
    flow_arrows: list[SymbolNode],
    img_gray: Optional[np.ndarray] = None,
    max_assign_dist_px: float = 200.0,
) -> Step7Result:
    """Tag contracted-graph edges with flow direction from arrow predictions.

    Modifies G in-place:
      "flow_direction" = "src_gid -> dst_gid"  (when img_gray provided)
      "has_arrow"      = True                   (when img_gray is None)

    The tip is the DOWNSTREAM end: whichever of the two adjacent symbol centroids
    is closer to the inferred arrowhead location receives flow.
    """
    n_assigned = 0
    n_directed = 0
    edges_directed: set[tuple[str, str]] = set()

    for arrow in flow_arrows:
        edge = _find_adjacent_edge(arrow.cx, arrow.cy, G, max_assign_dist_px)
        if edge is None:
            continue
        u_gid, v_gid = edge
        n_assigned += 1

        if img_gray is not None:
            tip_x, tip_y = _arrow_tip_xy(arrow, img_gray)
            ua, va = G.nodes[u_gid], G.nodes[v_gid]
            d_u = math.hypot(tip_x - ua.get("cx", 0.0), tip_y - ua.get("cy", 0.0))
            d_v = math.hypot(tip_x - va.get("cx", 0.0), tip_y - va.get("cy", 0.0))
            # Tip (arrowhead) is the downstream end — closer node is the destination
            src_gid, dst_gid = (v_gid, u_gid) if d_u <= d_v else (u_gid, v_gid)
            G.edges[u_gid, v_gid]["flow_direction"] = f"{src_gid} -> {dst_gid}"
            n_directed += 1
        else:
            G.edges[u_gid, v_gid]["has_arrow"] = True

        edges_directed.add((min(u_gid, v_gid), max(u_gid, v_gid)))

    return Step7Result(
        n_arrows_total=len(flow_arrows),
        n_arrows_assigned=n_assigned,
        n_arrows_directed=n_directed,
        n_edges_directed=len(edges_directed),
    )
