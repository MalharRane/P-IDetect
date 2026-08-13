"""Phase 6 Tier 1 -- mechanical grounding check (docs/phase6_tier1_design.md sec 3.1, sec 2.3).

Four checks, all purely mechanical (regex/substring over the transcript, no second LLM
call judging its own grounding -- sec 3.1 is explicit that this must be cheap and
deterministic so it can't be talked out of flagging a real violation):

  1. Citation check: every sym_<n>/junc_<n>-shaped token in the final answer text must
     have appeared somewhere in a tool result returned during this turn.
  2. Query-argument check (sec 2.3): every tag_function/tag_contains value passed to
     find_nodes/count_nodes must trace back to a value already present in an earlier
     list_systems_or_lines() result this turn -- closes "grounded facts, hallucinated query."
  3. Filter-completeness check (cross-sheet audit, docs/phase6_tier1_design/stress_results.md
     S5/C3/C4): when the question references a real cls_name on this sheet, absence/presence
     cannot be concluded without a find_nodes/count_nodes call that actually used cls_name=
     as an argument -- and if the question ALSO references a real tag_function or a quoted
     substring, that must be combined in the SAME call, not dropped. This is what makes the
     structural fix real: agent.py treats a violation here as a REJECTED answer (forces a
     retry with a corrective message), not just a flag for reporting after the fact.

     node_type is deliberately excluded from what must be combined with cls_name: in this
     schema a cls_name value belongs to exactly one node_type by construction, so requiring
     node_type IN ADDITION to a correct cls_name= call would reject already-sufficient
     evidence (e.g. count_nodes(cls_name='valve_handwheel') alone is a complete answer --
     'valve_handwheel' can only ever be a valve). tag_function/tag_contains are NOT
     redundant with cls_name the same way -- a subtype can have members with different tags
     -- which is why those, and only those, must be verified in the SAME call.
  4. Judgment-scope check (cross-sheet audit, sheet 8 K4): a question asking for a
     subjective/evaluative conclusion ("well-instrumented?", "most critical?", "safe?") is
     never answerable by any tool -- no tool returns a quality/criticality/safety field. An
     answer that renders a verdict adjective for such a question is a violation EVEN WHEN
     every number in it is genuinely grounded, because the citation/argument checks above
     only ever verify that facts are real, never that the conclusion drawn from them was
     itself licensed by a tool result. agent.py's classifier (is_judgment_bait, same idea as
     its traversal-bait classifier) is the primary defense -- it intercepts these questions
     before the LLM ever runs, so this check should rarely fire in practice. It stays here as
     a second, independent layer so the rule doesn't silently stop applying if that
     classifier's prompt/pattern ever changes and lets a judgment question through to a
     free-form answer.

All operate on a ToolCallRecord transcript, not on prose -- see agent.py for how it's built.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_ID_PATTERN = re.compile(r"\b(?:sym|junc)_\d+\b")
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_QUOTED_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")

_NODE_TYPE_WORDS = {
    "valve": "valve", "valves": "valve",
    "instrument": "instrument", "instruments": "instrument",
    "vessel": "vessel", "vessels": "vessel",
    "flow_arrow": "flow_arrow", "flow_arrows": "flow_arrow",
    "off_page": "off_page", "off-page": "off_page",
    "unknown_fitting": "unknown_fitting", "unknown_fittings": "unknown_fitting",
}

# Question forms asking for a subjective/evaluative conclusion about the sheet rather than a
# retrievable fact -- "well-instrumented", "most critical", "is X safe/adequate/good design".
# Deliberately scoped to these forms (not a general sentiment classifier) to keep false-positive
# risk low against the rest of the fixture vocabulary -- verified empirically against all three
# sheets' locked fixtures before use (no other question in any fixture matches this pattern).
_JUDGMENT_BAIT_PATTERN = re.compile(
    r"\bwell[- ]instrumented\b|\bwell[- ]designed\b|\bgood design\b|\bpoorly (?:instrumented|designed)\b|"
    r"\bmost (?:important|critical)\b|\b(?:over|under)[- ]instrumented\b|\bbest practice\b|"
    r"\b(?:is|are|does|do)\b.{0,60}\b(?:safe|adequate|sufficient|efficient|reliable|robust|risky|"
    r"well[- ]designed|well[- ]instrumented|good design|properly (?:designed|instrumented))\b",
    re.IGNORECASE,
)

# Verdict-adjective vocabulary that can never be licensed by a tool result -- no tool returns a
# quality/criticality/safety/adequacy field, so any of these words in an answer to a
# judgment-bait question is an unsupported conclusion regardless of what real numbers surround it.
_VERDICT_PATTERN = re.compile(
    r"\b(?:modest|adequate|inadequate|sufficient|insufficient|excellent|poor|poorly|"
    r"well[- ]instrumented|well[- ]designed|under[- ]instrumented|over[- ]instrumented|"
    r"robust|fragile|reliable|unreliable|efficient|inefficient|optimal|suboptimal|"
    r"safe|unsafe|risky|critical|(?:most|least) important|strong design|weak design)\b",
    re.IGNORECASE,
)


def is_judgment_bait(question_text: str) -> bool:
    return bool(_JUDGMENT_BAIT_PATTERN.search(question_text))


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
    filter_violations: list[str] = field(default_factory=list)
    judgment_violations: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        out = [f"uncited id in answer: {i}" for i in self.uncited_ids]
        out += self.ungrounded_queries
        out += self.filter_violations
        out += self.judgment_violations
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


def expected_filters(question_text: str, gi) -> dict[str, str]:
    """Extracts the filter axes a question plausibly specifies, each only when unambiguous --
    an axis with zero or multiple distinct matches is left out rather than guessed. `gi` is a
    GraphIndex; only used to look up the sheet's REAL cls_name/tag_function vocabulary, so this
    never depends on a fixed/hardcoded list that could go stale on a different sheet."""
    words = _WORD_PATTERN.findall(question_text)
    lower_words = [w.lower() for w in words]
    expected: dict[str, str] = {}

    node_type_matches = {_NODE_TYPE_WORDS[w] for w in lower_words if w in _NODE_TYPE_WORDS}
    if len(node_type_matches) == 1:
        expected["node_type"] = next(iter(node_type_matches))

    real_cls_names = {n["cls_name"] for n in gi.nodes_by_id.values() if n.get("cls_name")}
    cls_matches = {w for w in words if w in real_cls_names}
    if len(cls_matches) == 1:
        expected["cls_name"] = next(iter(cls_matches))

    real_tag_functions = {
        n["tag"]["function"] for n in gi.nodes_by_id.values()
        if n.get("tag") and n["tag"].get("function")
    }
    func_matches = {w for w in words if w in real_tag_functions}
    if len(func_matches) == 1:
        expected["tag_function"] = next(iter(func_matches))

    quoted = _QUOTED_PATTERN.findall(question_text)
    if len(quoted) == 1:
        expected["tag_contains"] = quoted[0]

    return expected


def _call_satisfies(call: ToolCallRecord, required: dict[str, str]) -> bool:
    if call.name not in ("find_nodes", "count_nodes"):
        return False
    for axis, value in required.items():
        if axis == "tag_contains":
            if value.lower() not in str(call.args.get("tag_contains", "")).lower():
                return False
        elif call.args.get(axis) != value:
            return False
    return True


def required_filters(expected: dict[str, str]) -> dict[str, str]:
    """The subset of expected_filters() that must all appear on the SAME find_nodes/count_nodes
    call to count as a complete, satisfying query. node_type is dropped whenever cls_name is
    also present -- redundant by schema construction, see module docstring's check-3 note --
    otherwise node_type is kept since it may be the only filter the question specified at all
    (e.g. a plain node_type + tag_contains question with no cls_name in play). Shared by
    check_filter_completeness below AND agent.py's post-call nudge/termination-guard, so both
    use the identical definition of "this call is already a complete answer" -- they used to
    diverge (the nudge checked the unfiltered dict, including node_type), which is exactly why
    a real find_nodes(cls_name=..., tag_function=...) call that fully answered the question
    never triggered the nudge: it had no node_type argument, since node_type is never needed
    once cls_name is known."""
    if "cls_name" in expected:
        return {k: v for k, v in expected.items() if k in ("cls_name", "tag_function", "tag_contains")}
    return dict(expected)


def check_filter_completeness(question_text: str, transcript: list[ToolCallRecord], gi) -> list[str]:
    """Rejects an answer built on a narrower query than the question specifies, whenever the
    question references a real cls_name (see module docstring for why cls_name is the trigger
    and node_type is deliberately excluded from what must be combined with it)."""
    expected = expected_filters(question_text, gi)
    if "cls_name" not in expected:
        return []  # this check only fires on cls_name-referencing questions -- see docstring

    required = required_filters(expected)
    if any(_call_satisfies(c, required) for c in transcript):
        return []

    return [
        f"question references cls_name={expected['cls_name']!r} "
        + ("".join(f"and {k}={v!r} " for k, v in required.items() if k != "cls_name"))
        + "but no single find_nodes/count_nodes call combined all of them -- absence or "
          "presence cannot be concluded from a narrower query (e.g. list_systems_or_lines() "
          "alone, or cls_name without the tag constraint)"
    ]


def check_judgment_scope(question_text: str, answer_text: str) -> list[str]:
    """See module docstring, check 4. Independent second layer behind agent.py's
    is_judgment_bait-driven classifier refusal -- fires only if a judgment question somehow
    reached a free-form answer anyway."""
    if not is_judgment_bait(question_text):
        return []
    hit = _VERDICT_PATTERN.search(answer_text)
    if not hit:
        return []
    return [
        f"answer renders an unsupported evaluative verdict ({hit.group(0)!r}) for a "
        f"subjective/judgment question -- no tool provides a quality/criticality/safety "
        f"field; report only the raw grounded data and decline the judgment itself"
    ]


def check_grounding(
    answer_text: str, transcript: list[ToolCallRecord], question_text: str = "", gi=None,
) -> GroundingResult:
    uncited_ids, ungrounded_queries = check_citations(answer_text, transcript, question_text)
    filter_violations = check_filter_completeness(question_text, transcript, gi) if gi is not None else []
    judgment_violations = check_judgment_scope(question_text, answer_text)
    return GroundingResult(
        ok=not uncited_ids and not ungrounded_queries and not filter_violations and not judgment_violations,
        uncited_ids=uncited_ids,
        ungrounded_queries=ungrounded_queries,
        filter_violations=filter_violations,
        judgment_violations=judgment_violations,
    )
