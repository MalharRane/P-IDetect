"""Phase 6 Tier 1 -- eval fixture builder (docs/phase6_tier1_design.md sec 4).

Computes all ~19 expected answers PROGRAMMATICALLY from the contracted graph JSON
(docs/phase4_step0_3/sheet_0_graph.json, regenerated with --run-ocr so instrument
tags are populated) -- no LLM involved anywhere, per sec 4.1's ground-truth-
independence rule. Locks the result to docs/phase6_tier1_design/sheet0_eval_expected.json.

Usage:
    python scripts/build_phase6_eval_fixture.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.agent.graph_index import provenance_of as _provenance  # single source of truth, sec 3.2

GRAPH_PATH = REPO / "docs" / "phase4_step0_3" / "sheet_0_graph.json"
OUT_PATH = REPO / "docs" / "phase6_tier1_design" / "sheet0_eval_expected.json"


def build(graph: dict) -> list[dict]:
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    adjacency: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for e in edges:
        prov = _provenance(e.get("branch_type", ""))
        adjacency[e["u"]].append((e["v"], {**e, "provenance": prov}))
        adjacency[e["v"]].append((e["u"], {**e, "provenance": prov}))

    type_counts = Counter(n["node_type"] for n in nodes.values())

    instrument_tags = [
        {"id": n["id"], **{k: n["tag"].get(k) for k in ("function", "loop_number", "raw_text", "parse_status")}}
        for n in nodes.values()
        if n["node_type"] == "instrument" and n.get("tag")
    ]
    parse_status_counts = Counter(t["parse_status"] for t in instrument_tags)
    ok_statuses = {"ok", "ok_placeholder", "single_line_split", "single_line_unsplit"}

    def neighbor_set(nid: str) -> dict[str, str]:
        return {v: e["provenance"] for v, e in adjacency[nid]}

    sym0_neighbors = neighbor_set("sym_0")

    control_valves = sorted(
        (n["id"] for n in nodes.values() if n.get("cls_name") == "control_valve_diaphragm"),
        key=lambda s: int(s.split("_")[1]),
    )
    vessels = sorted(
        (n["id"] for n in nodes.values() if n["node_type"] == "vessel"),
        key=lambda s: int(s.split("_")[1]),
    )
    pt_instruments = sorted(
        (t["id"] for t in instrument_tags if t["function"] == "PT"),
        key=lambda s: int(s.split("_")[1]),
    )
    substr_1401 = sorted(
        (t["id"] for t in instrument_tags if t["raw_text"] and "1401" in t["raw_text"]),
        key=lambda s: int(s.split("_")[1]),
    )
    ok_ids = sorted(
        (t["id"] for t in instrument_tags if t["parse_status"] in ok_statuses),
        key=lambda s: int(s.split("_")[1]),
    )
    failed_ids = sorted(
        (t["id"] for t in instrument_tags if t["parse_status"] not in ok_statuses),
        key=lambda s: int(s.split("_")[1]),
    )
    off_page_cls_names = sorted({n["cls_name"] for n in nodes.values() if n["node_type"] == "off_page"})
    existing_pt_loop_numbers = sorted(t["loop_number"] for t in instrument_tags if t["function"] == "PT")
    tanks = sorted(
        (n["id"] for n in nodes.values() if n.get("cls_name") == "tank"),
        key=lambda s: int(s.split("_")[1]),
    )

    assert "sym_999" not in nodes
    assert sum(1 for n in nodes.values() if n.get("cls_name") == "pump") == 0
    assert "sym_2" not in sym0_neighbors
    assert not any(t["raw_text"] == "PT 14099" for t in instrument_tags)  # guaranteed-absent trap tag
    # "tank" and "pump" are both cls_name subtypes of node_type=="vessel" -- a genuinely-present
    # counterpart to Q18's genuinely-absent "pump", so the fix (query cls_name, don't infer
    # absence/presence from the node_type breakdown) is proven both directions, not just tuned
    # to get "0 pumps" right by luck.
    assert len(tanks) >= 1

    questions = [
        dict(id=1, expected_kind="found",
             question="How many valves are on this sheet?",
             tools=["count_nodes(node_type='valve')"],
             expected={"count": type_counts["valve"]}),
        dict(id=2, expected_kind="found",
             question="How many nodes are on this sheet in total, broken down by type?",
             tools=["count_nodes()"],
             expected={"count": sum(type_counts.values()), "breakdown_by_node_type": dict(type_counts)}),
        dict(id=3, expected_kind="found",
             question="List all vessel/tank nodes.",
             tools=["find_nodes(node_type='vessel')"],
             expected={"ids": vessels}),
        dict(id=4, expected_kind="found",
             question="Find every control_valve_diaphragm on this sheet.",
             tools=["find_nodes(cls_name='control_valve_diaphragm')"],
             expected={"ids": control_valves}),
        dict(id=5, expected_kind="found",
             question="What is sym_0?",
             tools=["get_node('sym_0')"],
             expected={"node_type": nodes["sym_0"]["node_type"], "cls_name": nodes["sym_0"]["cls_name"],
                       "bbox": nodes["sym_0"]["bbox"]}),
        dict(id=6, expected_kind="found",
             question="What are the details of sym_113?",
             tools=["get_node('sym_113')"],
             expected={"node_type": nodes["sym_113"]["node_type"], "cls_name": nodes["sym_113"]["cls_name"]}),
        dict(id=7, expected_kind="found",
             question="What's connected to sym_0?",
             tools=["get_neighbors('sym_0')"],
             expected={"neighbor_ids": sorted(sym0_neighbors)}),
        dict(id=8, expected_kind="found",
             question="Is the connection between sym_0 and sym_74 a traced line or inferred?",
             tools=["get_neighbors('sym_0')"],
             expected={"provenance": sym0_neighbors.get("sym_74")}),
        dict(id=9, expected_kind="found",
             question="Is the connection between sym_0 and sym_79 a traced line or inferred?",
             tools=["get_neighbors('sym_0')"],
             expected={"provenance": sym0_neighbors.get("sym_79")}),
        dict(id=10, expected_kind="not_found",
             question="Does sym_0 connect directly to sym_2?",
             tools=["get_neighbors('sym_0')"],
             expected={"directly_connected": "sym_2" in sym0_neighbors}),
        dict(id=11, expected_kind="found",
             question="What is the OCR'd tag on sym_89?",
             tools=["get_node('sym_89')"],
             expected={"tag": nodes["sym_89"]["tag"]}),
        dict(id=12, expected_kind="found",
             question="Find all pressure transmitters (PT) on this sheet.",
             tools=["list_systems_or_lines()", "find_nodes(tag_function='PT')"],
             expected={"ids": pt_instruments}),
        dict(id=13, expected_kind="found",
             question="Find any instrument whose tag mentions '1401'.",
             tools=["list_systems_or_lines()", "find_nodes(tag_contains='1401')"],
             expected={"ids": substr_1401}),
        dict(id=14, expected_kind="found",
             question="How many instruments have a successfully-read tag vs. a failed OCR read?",
             tools=["count_nodes / find_nodes over parse_status"],
             expected={"ok_count": len(ok_ids), "failed_count": len(failed_ids),
                       "ok_ids": ok_ids, "failed_ids": failed_ids,
                       "failed_raw_text": {t["id"]: t["raw_text"] for t in instrument_tags
                                            if t["parse_status"] not in ok_statuses}},
             scoring_note=(
                 "sym_100/sym_101 are real instrument nodes whose OCR parse failed -- "
                 "function/loop_number are None, raw_text is the garbled literal OCR read "
                 "('PSV 1407 4', '07 PSV 1406'). A pass requires the agent to report these as "
                 "present with an unparsed/garbled tag (optionally quoting raw_text verbatim if "
                 "asked); the agent must NOT be marked wrong for not producing a 'corrected' "
                 "PSV-1407/PSV-1406-style tag, and must NOT be credited for fabricating one -- "
                 "faithfully surfacing the real graph state (including its OCR noise) is the "
                 "only correct behavior here, per sec 0's core principle.")),
        dict(id=15, expected_kind="found",
             question="What loop numbers / off-page connectors appear on this sheet?",
             tools=["list_systems_or_lines()"],
             expected={"instrument_loops": sorted(
                           ({"function": t["function"], "loop_number": t["loop_number"]}
                            for t in instrument_tags if t["function"] and t["loop_number"]),
                           key=lambda d: (d["function"], d["loop_number"]),
                       ),
                       "off_page_connectors": off_page_cls_names}),
        dict(id=16, expected_kind="not_found",
             question="What is sym_999?",
             tools=["get_node('sym_999')"],
             expected={"error": "not_found", "id": "sym_999"}),
        dict(id=17, expected_kind="refusal_required",
             question="Based on typical P&ID conventions, what's usually downstream of a control valve like sym_0?",
             tools=[],
             expected={"must_refuse_general_knowledge": True}),
        dict(id=18, expected_kind="not_found",
             question="How many pumps are on this sheet?",
             tools=["find_nodes(cls_name='pump')", "count_nodes(cls_name='pump')"],
             expected={"count": sum(1 for n in nodes.values() if n.get("cls_name") == "pump")}),
        dict(id=19, expected_kind="not_found",
             question="What's connected to line PT 14099?",
             tools=["list_systems_or_lines()"],
             expected={"exists": False, "existing_pt_loop_numbers": existing_pt_loop_numbers}),
        dict(id=20, expected_kind="found",
             question="How many tanks are on this sheet?",
             tools=["find_nodes(cls_name='tank')", "count_nodes(cls_name='tank')"],
             expected={"count": len(tanks), "ids": tanks},
             scoring_note=(
                 "Paired with Q18 (pump, genuinely absent): 'tank' is likewise a cls_name "
                 "subtype under node_type=='vessel', not a node_type value itself, and it "
                 "genuinely EXISTS here (unlike pump). A pass requires querying cls_name='tank' "
                 "specifically -- reasoning from the node_type breakdown (which shows "
                 "'vessel: 2' with no subtype split) is not sufficient grounding for a count "
                 "or an id list, even though 'vessel: 2' happens to equal the right number "
                 "here by coincidence (both vessels are tanks) -- sec 0's core principle: the "
                 "argument must be grounded, not just the conclusion.")),
    ]

    assert len(questions) == 20
    return questions


def main() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = build(graph)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({
            "sheet_id": graph["sheet_id"],
            "source_graph": str(GRAPH_PATH.relative_to(REPO)),
            "notes": {
                "failed_parse_policy": (
                    "sym_100 and sym_101 are real instrument nodes with tag.parse_status=='failed' "
                    "(OCR ran but produced unparseable text: raw_text='PSV 1407 4' and "
                    "'07 PSV 1406' respectively; function/loop_number are None). Any question "
                    "touching these nodes scores the agent correct for faithfully reporting them "
                    "as present-but-unparsed / quoting the raw garbled text -- never for inventing "
                    "a corrected tag, and never penalized for not producing one. Only Q14 directly "
                    "surfaces them in this fixture (via failed_ids/failed_raw_text); Q12/Q13/Q15 "
                    "correctly exclude them (function/loop_number are None, so no tag_function or "
                    "list_systems_or_lines match; tag_contains='1401' doesn't match their raw_text "
                    "either); Q2's total instrument count (36) correctly includes them as real "
                    "detected instrument nodes regardless of OCR outcome."
                ),
            },
            "questions": questions,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(questions)} questions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
