"""Phase 6 Tier 1 -- the five retrieval tools (docs/phase6_tier1_design.md sec 2).

Every tool is a pure function of GraphIndex -- no I/O, no model calls, never raises for
"no match" (sec 5: honest "not found," not a crash). TOOL_SCHEMAS + TOOL_FUNCTIONS is the
registry src/pidetect/agent/agent.py dispatches through; llm.py's providers only ever see
TOOL_SCHEMAS (name/description/JSON-Schema parameters), never these Python functions.

_OK_PARSE_STATUSES (sec 2.2's list_systems_or_lines bullet, from src/pidetect/text/ocr.py):
tags counted as "read succeeded" for the tag/line grounding universe. A "failed" parse
(garbled OCR, e.g. sheet 0's sym_100/sym_101) is a real instrument node with no reliable
function/loop_number -- get_node/get_neighbors still surface it (raw_text included) but it's
excluded from list_systems_or_lines/find_nodes-hint-on-miss, which are meant to be a clean
enumeration a query can be grounded against, not a place to launder OCR noise into a
false-positive "tag that exists."
"""
from __future__ import annotations

from pidetect.agent.graph_index import GraphIndex

_OK_PARSE_STATUSES = {"ok", "ok_placeholder", "single_line_split", "single_line_unsplit"}


def _numeric_id_key(node_id: str) -> tuple[str, int]:
    """sym_9 before sym_10 -- lexicographic string sort gets this backwards."""
    prefix, _, num = node_id.rpartition("_")
    return (prefix, int(num)) if num.isdigit() else (prefix, -1)


def _tag_summary(node: dict) -> str | None:
    tag = node.get("tag")
    if not tag or not tag.get("function") or not tag.get("loop_number"):
        return None
    return f"{tag['function']}-{tag['loop_number']}"


def _light_summary(node: dict) -> dict:
    return {
        "id": node["id"],
        "node_type": node["node_type"],
        "cls_name": node["cls_name"],
        "tag_summary": _tag_summary(node),
    }


def _existing_function_codes(gi: GraphIndex) -> list[str]:
    codes = {
        n["tag"]["function"]
        for n in gi.nodes_by_id.values()
        if n["node_type"] == "instrument" and n.get("tag")
        and n["tag"].get("parse_status") in _OK_PARSE_STATUSES
        and n["tag"].get("function")
    }
    return sorted(codes)


def _existing_tag_texts(gi: GraphIndex) -> list[str]:
    texts = {
        n["tag"]["raw_text"]
        for n in gi.nodes_by_id.values()
        if n["node_type"] == "instrument" and n.get("tag")
        and n["tag"].get("parse_status") in _OK_PARSE_STATUSES
        and n["tag"].get("raw_text")
    }
    return sorted(texts)


# ---------------------------------------------------------------------------
# find_nodes
# ---------------------------------------------------------------------------

def find_nodes(
    gi: GraphIndex,
    node_type: str | None = None,
    cls_name: str | None = None,
    tag_function: str | None = None,
    tag_contains: str | None = None,
) -> dict:
    matches = []
    for n in gi.nodes_by_id.values():
        if node_type is not None and n["node_type"] != node_type:
            continue
        if cls_name is not None and n["cls_name"] != cls_name:
            continue
        if tag_function is not None:
            tag = n.get("tag")
            if not tag or tag.get("function") != tag_function:
                continue
        if tag_contains is not None:
            tag = n.get("tag")
            raw = (tag or {}).get("raw_text") or ""
            if tag_contains.lower() not in raw.lower():
                continue
        matches.append(n)

    matches.sort(key=lambda n: _numeric_id_key(n["id"]))
    result = [_light_summary(n) for n in matches]

    if result:
        return {"matches": result}

    # docs/phase6_tier1_design.md sec 2.2: a tag-filtered miss is never a bare [] -- it
    # carries the real existing set so the caller (or the grounding check, sec 2.3) always
    # has something to fall back on even if list_systems_or_lines() was never called.
    if tag_function is not None:
        return {"matches": [], "requested_function": tag_function,
                "existing_function_codes": _existing_function_codes(gi)}
    if tag_contains is not None:
        return {"matches": [], "requested_substring": tag_contains,
                "existing_tag_texts": _existing_tag_texts(gi)}
    return {"matches": []}


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

def get_node(gi: GraphIndex, id: str) -> dict:
    node = gi.nodes_by_id.get(id)
    if node is None:
        return {"error": "not_found", "id": id}
    return dict(node)


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

def get_neighbors(gi: GraphIndex, id: str) -> dict | list[dict]:
    if id not in gi.nodes_by_id:
        return {"error": "not_found", "id": id}

    out = []
    for neighbor_id, edata in gi.adjacency.get(id, []):
        nbr = gi.nodes_by_id.get(neighbor_id, {})
        out.append({
            "neighbor_id": neighbor_id,
            "neighbor_node_type": nbr.get("node_type"),
            "neighbor_cls_name": nbr.get("cls_name"),
            "neighbor_tag_summary": _tag_summary(nbr) if nbr else None,
            "branch_type": edata.get("branch_type"),
            "provenance": edata["provenance"],
            "flow_direction": edata.get("flow_direction"),
            "length_px": edata.get("length_px"),
        })
    out.sort(key=lambda d: _numeric_id_key(d["neighbor_id"]))
    return out


# ---------------------------------------------------------------------------
# count_nodes
# ---------------------------------------------------------------------------

def count_nodes(
    gi: GraphIndex,
    node_type: str | None = None,
    cls_name: str | None = None,
    tag_function: str | None = None,
) -> dict:
    matched = [
        n for n in gi.nodes_by_id.values()
        if (node_type is None or n["node_type"] == node_type)
        and (cls_name is None or n["cls_name"] == cls_name)
        and (tag_function is None or (n.get("tag") and n["tag"].get("function") == tag_function))
    ]
    if node_type is None and cls_name is None and tag_function is None:
        breakdown: dict[str, int] = {}
        for n in gi.nodes_by_id.values():
            breakdown[n["node_type"]] = breakdown.get(n["node_type"], 0) + 1
        return {"count": len(matched), "breakdown_by_node_type": breakdown}
    return {"count": len(matched)}


# ---------------------------------------------------------------------------
# list_systems_or_lines
# ---------------------------------------------------------------------------

def list_systems_or_lines(gi: GraphIndex) -> dict:
    instrument_tags = []
    for n in gi.nodes_by_id.values():
        if n["node_type"] != "instrument":
            continue
        tag = n.get("tag")
        if not tag or tag.get("parse_status") not in _OK_PARSE_STATUSES:
            continue
        instrument_tags.append({
            "id": n["id"],
            "function": tag.get("function"),
            "loop_number": tag.get("loop_number"),
            "raw_text": tag.get("raw_text"),
        })
    instrument_tags.sort(key=lambda t: _numeric_id_key(t["id"]))

    off_page_connectors = sorted({
        n["cls_name"] for n in gi.nodes_by_id.values() if n["node_type"] == "off_page"
    })

    return {"instrument_tags": instrument_tags, "off_page_connectors": off_page_connectors}


TOOL_FUNCTIONS = {
    "find_nodes": find_nodes,
    "get_node": get_node,
    "get_neighbors": get_neighbors,
    "count_nodes": count_nodes,
    "list_systems_or_lines": list_systems_or_lines,
}

TOOL_SCHEMAS = [
    {
        "name": "find_nodes",
        "description": (
            "Search sheet nodes by type, class, or instrument tag. tag_function/tag_contains "
            "must be values already seen in a prior list_systems_or_lines() result this turn -- "
            "never a guessed tag/line name. Returns {'matches': [...]}; a tag-filtered miss also "
            "returns the real existing set so a wrong guess never comes back as a bare empty list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string",
                              "enum": ["valve", "instrument", "unknown_fitting", "flow_arrow",
                                       "off_page", "vessel"]},
                "cls_name": {"type": "string", "description": "exact detected class name, e.g. 'control_valve_diaphragm'"},
                "tag_function": {"type": "string", "description": "exact ISA function code, e.g. 'PT' -- must come from list_systems_or_lines()"},
                "tag_contains": {"type": "string", "description": "case-insensitive substring of an instrument tag's raw_text -- must come from list_systems_or_lines()"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_node",
        "description": "Full detail for one node id. Returns {'error':'not_found','id':...} if it doesn't exist -- never guess.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_neighbors",
        "description": (
            "Direct (1-hop only) connections of one node id, each tagged 'provenance': "
            "'TRACED' (a single continuously-traced skeleton run) or 'INFERRED' (a bridged "
            "gap or a junction-contraction reroute -- weaker evidence). No multi-hop traversal exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "count_nodes",
        "description": "Count nodes matching filters (same semantics as find_nodes, no tag_contains). No filters -> total count plus a breakdown by node_type.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string",
                              "enum": ["valve", "instrument", "unknown_fitting", "flow_arrow",
                                       "off_page", "vessel"]},
                "cls_name": {"type": "string"},
                "tag_function": {"type": "string", "description": "must come from list_systems_or_lines()"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_systems_or_lines",
        "description": (
            "The canonical ground-truth enumeration of every instrument tag (id/function/"
            "loop_number/raw_text) and off-page connector label on this sheet. Call this "
            "BEFORE any find_nodes/count_nodes call with tag_function or tag_contains -- "
            "never guess a tag or line name, look it up here first. Note: this lists instrument "
            "loop identities and off-page labels, not parsed pipe line numbers (not extracted by this pipeline)."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]
