"""Phase 4 step 8: export the contracted graph to JSON and PID2Graph-compatible GraphML.

JSON format  — human-readable / API-friendly:
  {
    "sheet_id": N,
    "nodes": [{"id", "node_type", "cls_name", "bbox":[x1,y1,x2,y2], "conf"}, ...],
    "edges": [{"u", "v", "flow_direction"?}, ...]
  }

GraphML format — PID2Graph-compatible for diff / evaluate.py:
  Node attributes: label (string), xmin, ymin, xmax, ymax (float)
  Edge attributes: edge_label = "solid"  (+ optional flow_direction string)

Public API:
    export_json(G, sheet_id, out_path)
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
    return {
        "id":        gid,
        "node_type": attrs.get("node_type", ""),
        "cls_name":  attrs.get("cls_name", ""),
        "bbox": [
            attrs.get("x1", 0.0), attrs.get("y1", 0.0),
            attrs.get("x2", 0.0), attrs.get("y2", 0.0),
        ],
        "conf": round(float(attrs.get("conf", 0.0)), 4),
    }


def _edge_to_dict(u: str, v: str, attrs: dict) -> dict:
    d: dict = {"u": u, "v": v}
    if "flow_direction" in attrs:
        d["flow_direction"] = attrs["flow_direction"]
    elif attrs.get("has_arrow"):
        d["has_arrow"] = True
    # Carry numeric attributes that survive contraction
    for key in ("branch_id", "length_px", "branch_type"):
        if key in attrs:
            d[key] = attrs[key]
    return d


def export_json(G: nx.Graph, sheet_id: int | str, out_path: Path) -> None:
    """Write contracted graph to JSON (numpy-safe: no coords_rc)."""
    nodes = [_node_to_dict(gid, a) for gid, a in G.nodes(data=True)]
    edges = [_edge_to_dict(u, v, a) for u, v, a in G.edges(data=True)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"sheet_id": sheet_id, "nodes": nodes, "edges": edges}, indent=2),
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
        G_out.add_node(
            gid,
            label=_graphml_label(attrs),
            xmin=float(attrs.get("x1", 0.0)),
            ymin=float(attrs.get("y1", 0.0)),
            xmax=float(attrs.get("x2", 0.0)),
            ymax=float(attrs.get("y2", 0.0)),
        )

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
