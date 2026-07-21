"""Phase 4 step 8: export the contracted graph to JSON and PID2Graph-compatible GraphML.

JSON format  — human-readable / API-friendly:
  {
    "sheet_id": N,
    "nodes": [{"id", "node_type", "cls_name", "bbox":[x1,y1,x2,y2], "conf", "tag"?}, ...],
    "edges": [{"u", "v", "flow_direction"?, "branch_type"?, "path"?}, ...]
  }
  "tag" (instrument nodes that went through OCR only): {function, loop_number, raw_text,
    confidence, parse_status}.
  "branch_type" (string, docs/phase5_design.md): "direct" | "short_gap" | "connector" |
    "crossing" | "tip_to_junction" | "junction_to_junction" -- see _BRANCH_TYPE_LABELS.
  "path" (Phase 5 API only, via pidetect.pipeline -> graph_to_dict): downsampled
    [[x,y], ...] polyline of the traced skeleton branch, when one exists for that edge
    (absent for short-gap and contracted-through edges -- frontend draws a straight
    segment for those).

GraphML format — PID2Graph-compatible for diff / evaluate.py:
  Node attributes: label (string), xmin, ymin, xmax, ymax (float)
  Edge attributes: edge_label = "solid"  (+ optional flow_direction string)

Public API:
    export_json(G, sheet_id, out_path)
    graph_to_dict(G, sheet_id) -> dict   (in-memory, same shape export_json writes)
    export_graphml(G, out_path)
    run_step8(G, sheet_id, out_dir) -> dict[str, Path]
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Node label mapping: our node_type -> nearest PID2Graph GT label
# ---------------------------------------------------------------------------

def _graphml_label(attrs: dict) -> str:
    """Return the PID2Graph-compatible label for one graph node."""
    node_type = attrs.get("node_type", "")
    if node_type == "vessel":
        return attrs.get("cls_name", "tank")   # "tank" or "pump" from GT
    return {
        "valve":           "valve",
        "instrument":      "instrumentation",
        "unknown_fitting": "unknown_fitting",
        "off_page":        "inlet/outlet",
    }.get(node_type, node_type)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _node_to_dict(gid: str, attrs: dict) -> dict:
    d = {
        "id":        gid,
        "node_type": attrs.get("node_type", ""),
        "cls_name":  attrs.get("cls_name", ""),
        "bbox": [
            attrs.get("x1", 0.0), attrs.get("y1", 0.0),
            attrs.get("x2", 0.0), attrs.get("y2", 0.0),
        ],
        "conf": round(float(attrs.get("conf", 0.0)), 4),
    }
    # Phase 3 (docs/phase3_design.md §5): present only for instrument nodes that
    # actually went through OCR (tag_parse_status != "not_run"/absent).
    if attrs.get("node_type") == "instrument" and attrs.get("tag_parse_status"):
        d["tag"] = {
            "function":    attrs.get("tag_function"),
            "loop_number": attrs.get("tag_loop_number"),
            "raw_text":    attrs.get("tag_raw_text", ""),
            "confidence":  round(float(attrs.get("tag_confidence", 0.0)), 4),
            "parse_status": attrs.get("tag_parse_status"),
        }
    return d


# branch_type (pidetect.graph.lines.Branch): 0=tip-tip, 1=tip-junc, 2=junc-junc, 3=cycle;
# build.py's short-gap edges use -1. Only branch_type 0 ("direct") and -1 ("short_gap")
# ever survive step-6 contraction unchanged (tip-junc/junc-junc edges are removed along
# with the junction node they touch, replaced by a synthesized contracted edge instead) --
# kept complete here anyway so the mapping doesn't silently go stale if that changes.
_BRANCH_TYPE_LABELS: dict[int, str] = {
    -1: "short_gap",
    0:  "direct",
    1:  "tip_to_junction",
    2:  "junction_to_junction",
    3:  "cycle",
}

# Cap on polyline points per edge (Phase 5 API, docs/phase5_design.md): downsampling is
# adaptive (stride = branch_length // cap) rather than a fixed pixel stride, so one very
# long pipe run doesn't dominate payload size while short branches keep enough points to
# look like a real traced path rather than a straight line.
_MAX_PATH_POINTS = 40


def _downsample_path(coords_rc) -> list[list[float]]:
    """(row, col) skeleton polyline -> downsampled [[x, y], ...], endpoints always kept."""
    n = len(coords_rc)
    if n == 0:
        return []
    stride = max(1, n // _MAX_PATH_POINTS)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [[float(coords_rc[i][1]), float(coords_rc[i][0])] for i in idx]


def _edge_to_dict(u: str, v: str, attrs: dict) -> dict:
    d: dict = {"u": u, "v": v}
    if "flow_direction" in attrs:
        d["flow_direction"] = attrs["flow_direction"]
    elif attrs.get("has_arrow"):
        d["has_arrow"] = True
    if "branch_id" in attrs:
        d["branch_id"] = attrs["branch_id"]
    if "length_px" in attrs:
        d["length_px"] = attrs["length_px"]
    if "branch_type" in attrs:
        d["branch_type"] = _BRANCH_TYPE_LABELS.get(attrs["branch_type"], str(attrs["branch_type"]))
    elif attrs.get("via_type"):
        # Contracted-through edge (step 6): no single traced branch, but via_type
        # ("connector" | "crossing") is already a meaningful string -- surface it so
        # every edge carries some type label, not just uncontracted direct/short-gap ones.
        d["branch_type"] = attrs["via_type"]
    # Frontend (docs/phase5_design.md): draw the actual pipe route, not a straight
    # centroid-to-centroid chord. Short-gap and contracted-through edges have no single
    # traced branch -- "path" is simply absent and the frontend draws a straight segment.
    if "coords_rc" in attrs:
        d["path"] = _downsample_path(attrs["coords_rc"])
    return d


def graph_to_dict(G: nx.Graph, sheet_id: int | str) -> dict:
    """Same shape as export_json() writes, returned in-memory. Edges carry a
    downsampled "path" polyline when a traced skeleton branch is available (see
    _edge_to_dict) -- the only numpy structure (coords_rc) is converted to plain
    floats there, so the overall result stays JSON/numpy-safe.

    Used by the Phase 5 API (pidetect.pipeline) to hand a job's result straight to a
    JSON response without a round-trip through disk.
    """
    nodes = [_node_to_dict(gid, a) for gid, a in G.nodes(data=True)]
    edges = [_edge_to_dict(u, v, a) for u, v, a in G.edges(data=True)]
    return {"sheet_id": sheet_id, "nodes": nodes, "edges": edges}


def export_json(G: nx.Graph, sheet_id: int | str, out_path: Path) -> None:
    """Write contracted graph to JSON (numpy-safe: raw coords_rc arrays never touch json.dumps)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(graph_to_dict(G, sheet_id), indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# GraphML export
# ---------------------------------------------------------------------------

def export_graphml(G: nx.Graph, out_path: Path) -> None:
    """Write contracted graph to PID2Graph-compatible GraphML.

    Builds a clean copy of the graph with only the attributes that
    nx.write_graphml() can serialise (no numpy arrays, no lists).

    Edges carry edge_label="solid" for GT compatibility.  When a
    flow_direction is known it is written as a separate string attribute.
    """
    G_out = nx.Graph()

    for gid, attrs in G.nodes(data=True):
        node_attrs = dict(
            label=_graphml_label(attrs),
            xmin=float(attrs.get("x1", 0.0)),
            ymin=float(attrs.get("y1", 0.0)),
            xmax=float(attrs.get("x2", 0.0)),
            ymax=float(attrs.get("y2", 0.0)),
        )
        # Phase 3 (docs/phase3_design.md §5): extra string attrs, PID2Graph-compatible
        # readers ignore unknown attributes. None -> "" since GraphML has no null.
        if attrs.get("node_type") == "instrument" and attrs.get("tag_parse_status"):
            node_attrs.update(
                tag_function=attrs.get("tag_function") or "",
                tag_loop_number=attrs.get("tag_loop_number") or "",
                tag_raw_text=attrs.get("tag_raw_text", "") or "",
                tag_confidence=float(attrs.get("tag_confidence", 0.0)),
                tag_parse_status=attrs.get("tag_parse_status", ""),
            )
        G_out.add_node(gid, **node_attrs)

    for u, v, attrs in G.edges(data=True):
        eattrs: dict = {"edge_label": "solid"}
        fd = attrs.get("flow_direction")
        if fd:
            eattrs["flow_direction"] = fd
        G_out.add_edge(u, v, **eattrs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G_out, str(out_path))


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_step8(
    G: nx.Graph,
    sheet_id: int | str,
    out_dir: Path,
) -> dict[str, Path]:
    """Export contracted graph to JSON and GraphML. Returns {"json": path, "graphml": path}."""
    stem = f"sheet_{sheet_id}"
    json_path    = out_dir / f"{stem}_graph.json"
    graphml_path = out_dir / f"{stem}_graph.graphml"
    export_json(G, sheet_id, json_path)
    export_graphml(G, graphml_path)
    return {"json": json_path, "graphml": graphml_path}
