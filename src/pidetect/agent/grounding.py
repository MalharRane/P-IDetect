"""Phase 6 Tier 1 -- mechanical grounding check (docs/phase6_tier1_design.md sec 3.1, sec 2.3).

Two checks, both purely mechanical (regex/substring over the transcript, no second LLM
call judging its own grounding -- sec 3.1 is explicit that this must be cheap and
deterministic so it can't be talked out of flagging a real violation):

  1. Citation check: every sym_<n>/junc_<n>-shaped token in the final answer text must
     have appeared somewhere in a tool result returned during this turn.
  2. Query-argument check (sec 2.3): every tag_function/tag_contains value passed to
     find_nodes/count_nodes must trace back to a value already present in an earlier
     list_systems_or_lines() result this turn -- closes "grounded facts, hallucinated query."

Both operate on a ToolCallRecord transcript, not on prose -- see agent.py for how it's built.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_ID_PATTERN = re.compile(r"\b(?:sym|junc)_\d+\b")


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: object


@dataclass
class GroundingResult:
    ok: bool
    uncited_ids: list[str] = field(default_factory=list)
    ungrounded_queries: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        out = [f"uncited id in answer: {i}" for i in self.uncited_ids]
        out += self.ungrounded_queries
        return out


def _ids_in(obj) -> set[str]:
    return set(_ID_PATTERN.findall(json.dumps(obj, default=str)))


def check_citations(
    answer_text: str, transcript: list[ToolCallRecord], question_text: str = "",
) -> tuple[list[str], list[str]]:
    """Returns (uncited_ids, ungrounded_queries)."""
    answer_ids = set(_ID_PATTERN.findall(answer_text))

    # An id the USER themselves named (e.g. "does sym_0 connect to sym_2?") is legitimately
    # citable even if no tool result ever echoes it back -- correctly reporting "sym_2 is not
    # in sym_0's neighbor list" is the grounded, correct not_found answer (sec 4.4), not a
    # fabrication, even though "sym_2" only ever appears by its ABSENCE from a tool result.
    known_ids: set[str] = set(_ID_PATTERN.findall(question_text))
    list_systems_texts: set[str] = set()  # every string value seen in a list_systems_or_lines() result

    for call in transcript:
        # An id the agent looked UP (e.g. get_neighbors(id="sym_0")) is just as grounded as
        # one that came back in a result -- the agent legitimately queried it this turn, so
        # restating it in the answer isn't fabrication. Scanning only call.result (as this
        # originally did) false-flagged exactly this case: "what's connected to sym_0" answers
        # that (correctly) reference sym_0 itself were being marked as an uncited-id violation.
        known_ids |= _ids_in(call.args)
        known_ids |= _ids_in(call.result)
        if call.name == "list_systems_or_lines":
            for tag in call.result.get("instrument_tags", []):
                for v in (tag.get("function"), tag.get("loop_number"), tag.get("raw_text")):
                    if v:
                        list_systems_texts.add(str(v))
            list_systems_texts.update(call.result.get("off_page_connectors", []))

    uncited_ids = sorted(answer_ids - known_ids)

    ungrounded_queries: list[str] = []
    for call in transcript:
        if call.name not in ("find_nodes", "count_nodes"):
            continue
        for arg_name in ("tag_function", "tag_contains"):
            value = call.args.get(arg_name)
            if value is None:
                continue
            # tag_contains is a substring match -- ground it against list_systems_or_lines
            # by substring containment; tag_function must match a listed code exactly.
            grounded = (
                value in list_systems_texts
                if arg_name == "tag_function"
                else any(value.lower() in t.lower() for t in list_systems_texts)
            )
            if not grounded:
                ungrounded_queries.append(
                    f"{call.name}({arg_name}={value!r}) not traced to a prior "
                    f"list_systems_or_lines() result"
                )

    return uncited_ids, ungrounded_queries


def check_grounding(
    answer_text: str, transcript: list[ToolCallRecord], question_text: str = "",
) -> GroundingResult:
    uncited_ids, ungrounded_queries = check_citations(answer_text, transcript, question_text)
    return GroundingResult(
        ok=not uncited_ids and not ungrounded_queries,
        uncited_ids=uncited_ids,
        ungrounded_queries=ungrounded_queries,
    )
