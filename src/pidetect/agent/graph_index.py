"""Phase 6 Tier 1 -- in-memory index over one sheet's exported (contracted) graph JSON.

docs/phase6_tier1_design.md sec 2.1: tools operate on the graph_to_dict()/export_json()
shape (src/pidetect/graph/export.py), not the raw networkx.Graph -- that shape is already
the project's one stable, numpy-safe public contract. This module never touches pipeline
internals.

Public API:
    provenance_of(branch_type) -> "TRACED" | "INFERRED"   (sec 3.2's single source of truth --
        the eval fixture builder, scripts/build_phase6_eval_fixture.py, imports this too, so
        the two can never silently disagree)
    load_graph_index(source: dict) -> GraphIndex
    load_graph_index_from_file(path) -> GraphIndex
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

# docs/phase6_tier1_design.md sec 3.2: a single continuous skeleton polyline with zero
# contraction steps. Everything else (short_gap, connector, crossing) is INFERRED --
# see that section for why checking only the top-level branch_type/via_type string is
# sufficient (via_type/is_backbone both propagate through arbitrarily deep contraction
# chains in src/pidetect/graph/build.py, so a contracted edge can never masquerade as one
# of these three).
_TRACED_BRANCH_TYPES = {"direct", "tip_to_junction", "junction_to_junction"}


def provenance_of(branch_type: str) -> str:
    return "TRACED" if branch_type in _TRACED_BRANCH_TYPES else "INFERRED"


@dataclass
class GraphIndex:
    sheet_id: str
    nodes_by_id: dict[str, dict]
    # node id -> [(neighbor_id, edge_dict_with_provenance), ...], both directions of each edge
    adjacency: dict[str, list[tuple[str, dict]]]


def load_graph_index(source: dict) -> GraphIndex:
    nodes_by_id = {n["id"]: n for n in source["nodes"]}
    adjacency: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for e in source["edges"]:
        edata = {**e, "provenance": provenance_of(e.get("branch_type", ""))}
        adjacency[e["u"]].append((e["v"], edata))
        adjacency[e["v"]].append((e["u"], edata))
    return GraphIndex(
        sheet_id=source.get("sheet_id", ""),
        nodes_by_id=nodes_by_id,
        adjacency=dict(adjacency),
    )


def load_graph_index_from_file(path: str | Path) -> GraphIndex:
    return load_graph_index(json.loads(Path(path).read_text(encoding="utf-8")))
