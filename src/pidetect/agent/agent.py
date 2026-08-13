"""Phase 6 Tier 1 -- the agent loop (docs/phase6_tier1_design.md sec 1.4).

Owns orchestration only: call provider.step(); dispatch tool_call turns through the
registry (tools.py); run the mechanical grounding check (grounding.py) on the final
answer. No provider-specific code here (llm.py) and no retrieval logic here (tools.py) --
this module just wires the three together plus the round-cap/failure-handling policy
from sec 5.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pidetect.agent.graph_index import GraphIndex
from pidetect.agent.grounding import (
    GroundingResult, ToolCallRecord, check_grounding, expected_filters, is_judgment_bait,
    required_filters, _call_satisfies,
)
from pidetect.agent.llm import LLMProvider, Message, ProviderError, ToolSchema
from pidetect.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MAX_ROUNDS = 6  # sec 1.4: generous backstop, not an expected-path limit
MAX_GROUNDING_RETRIES = 2  # sec below: self-correction attempts before an honest non-answer

# Cross-sheet audit (T2/O2, docs/phase6_tier1_design/stress_results.md): "trace the pipe"-style
# questions repeatedly burned the whole round budget probing before giving up, because nothing
# stopped the model from TRYING multi-hop exploration. Recognition-then-refuse: classify these
# BEFORE any LLM call at all, so the answer is deterministic and costs zero rounds -- not a
# prompt instruction the model can forget partway through a long tool-call sequence.
_TRAVERSAL_BAIT_PATTERN = re.compile(
    r"\btrace\b|\bshortest path\b|\bpath\b|\broute\b|\bdownstream of\b|\bupstream of\b|"
    r"\bwould stop flowing\b|\bwere closed\b|\bwere shut\b",
    re.IGNORECASE,
)
_MENTIONED_ID_PATTERN = re.compile(r"\b(?:sym|junc)_\d+\b")

SYSTEM_PROMPT = """You are a grounded P&ID query agent. You answer questions about ONE \
piping-and-instrumentation diagram sheet using ONLY the tools provided -- find_nodes, \
get_node, get_neighbors, count_nodes, list_systems_or_lines, resolve_term.

CORE RULE (never violate this): you never answer from your own knowledge of P&IDs, ISA \
symbology, or piping conventions. Every id, tag, count, or connection you state must have \
appeared in a tool result you actually received this turn. If a fact is not in a tool \
result, it does not go in your answer. If a user asks what is "typically" or "usually" \
true of P&ID equipment in general (not about THIS sheet's actual data), refuse to answer \
from general knowledge and instead offer to look up the real data with a tool.

TAG/LINE QUERIES: never guess a tag or line name. Before calling find_nodes or count_nodes \
with tag_function or tag_contains, call list_systems_or_lines() first and pass only a value \
you saw in its result. If the user's requested tag/line does not appear there, say so \
plainly and name the tags that DO exist -- never a bare "not found" with no alternatives \
when you already have them.

NOT FOUND: an empty result, a missing node id, or a zero count is a completely valid, \
correct answer -- report it plainly. Never treat "the graph doesn't have this" as a \
reason to guess, retry with a different meaning, or fabricate something plausible.

node_type vs cls_name -- TWO SEPARATE AXES, DO NOT CONFLATE THEM: node_type is a small \
closed set (valve, instrument, unknown_fitting, flow_arrow, off_page, vessel) -- \
count_nodes() with no filter breaks totals down ONLY by node_type. cls_name is the specific \
detected subtype (e.g. "tank", "pump", "control_valve_diaphragm") and is INDEPENDENT of \
node_type -- several different cls_name values share one node_type (both "tank" and "pump" \
are node_type=vessel). A node_type breakdown can NEVER contain a cls_name value, whether or \
not that subtype exists on the sheet -- its absence from that breakdown is NOT evidence of \
anything. So: whenever the user asks about something that is not itself one of the six \
node_type values (a subtype like "pump", "tank", a specific valve class, etc.), you MUST \
query it with cls_name (find_nodes(cls_name=...) or count_nodes(cls_name=...)) before \
concluding it's absent -- inferring absence from "it's not in the node_type breakdown" is an \
ungrounded argument even when the conclusion happens to be true. list_systems_or_lines() only \
covers instrument TAGS -- a cls_name subtype will NEVER appear there even when it's real, so \
"I didn't see it in list_systems_or_lines()" is never valid evidence of absence for a subtype. \
If you're unsure whether a term is a node_type, a cls_name, or a tag, call resolve_term(term) \
to find out before answering. This is enforced mechanically, not just requested: an answer \
that references a real cls_name without a matching find_nodes/count_nodes(cls_name=...) call \
(combined with any tag constraint the question also specifies) will be rejected and you will \
be asked to correct it.

1 HOP ONLY: get_neighbors returns direct connections only. If a question needs more than \
one hop, say that plainly rather than chaining calls and presenting the result as a single \
verified fact. This applies even if you technically CAN answer it by calling get_neighbors \
twice (once on the original node, once on one of its neighbors) -- doing that and then \
describing the result as "A and B are connected through C" or any other single relationship \
is fabricating a claim no tool result actually made. Two 1-hop facts are two separate, \
individually-provenanced facts, never one verified path. State each hop and its own \
provenance separately, and say plainly that you have no way to confirm an actual route \
between the two beyond that.

PROVENANCE: get_neighbors labels each connection TRACED (a single continuously-traced \
skeleton run) or INFERRED (a bridged gap or a junction-contraction reroute -- weaker \
evidence). This is NOT decoration -- every time you state an INFERRED connection, say so in \
words that change the reader's confidence, not just the label. Weak: "connected via an \
INFERRED short gap." Correct: "connected, but only by an inferred short-gap bridge -- this \
was never traced as continuous ink, so verify it against the actual drawing before relying \
on it." Do this every time you cite an INFERRED connection, not only when the user explicitly \
asks how confident you are.

SPATIAL / PROXIMITY QUESTIONS: you have no tool that computes distance, adjacency, or "near" \
in any geometric sense -- node bboxes exist in the data but no tool exposes a distance \
computation over them, and pixel proximity on a P&ID does not reliably indicate a process \
relationship anyway (two symbols can sit close together on the page while belonging to \
unrelated lines, or far apart while directly piped together). If asked what's "near," \
"closest to," or spatially associated with something, say plainly that you cannot determine \
spatial proximity -- do not silently answer a different, nearby question (like "what's \
connected to X") instead without saying that's what you're doing.

JUDGMENT / EVALUATIVE QUESTIONS: no tool computes a design-quality, adequacy, safety, or \
importance/criticality assessment -- counts and connections are facts, they are not a \
verdict. If asked something evaluative (whether the sheet or its instrumentation is good, \
adequate, safe, or well designed, or which element is "most important"/"most critical"), you \
may report real counts or facts you actually retrieved, but you must NOT render the \
evaluative conclusion itself -- that judgment is not in the graph no matter how real the \
numbers around it are. State plainly that you cannot make that judgment call from graph data \
alone and offer the raw numbers instead, without restating the evaluative wording the user \
used. This is enforced mechanically: an evaluative verdict for this kind of question will be \
rejected even if every number in it is real.
"""


@dataclass
class AgentResult:
    text: str
    transcript: list[ToolCallRecord]
    grounding: GroundingResult
    rounds_used: int
    hit_round_cap: bool = False
    error: str | None = None
    refused_by_classifier: bool = False  # traversal-bait/judgment-bait short-circuit, zero LLM calls made
    grounding_retries: int = 0  # times a final_answer was rejected and retried this turn
    redundant_tool_calls_blocked: int = 0  # sec C4 fix: extra calls blocked after filters already satisfied


def _is_traversal_bait(question: str) -> bool:
    return bool(_TRAVERSAL_BAIT_PATTERN.search(question))


class Agent:
    def __init__(self, provider: LLMProvider, graph_index: GraphIndex, max_rounds: int = MAX_ROUNDS):
        self.provider = provider
        self.gi = graph_index
        self.max_rounds = max_rounds
        self.tool_schemas = [ToolSchema(**s) for s in TOOL_SCHEMAS]

    def _dispatch(self, name: str, args: dict) -> object:
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            return {"error": "invalid_tool", "name": name}
        try:
            return fn(self.gi, **args)
        except TypeError as exc:
            # sec 5: a schema-invalid call (bad enum value, unknown kwarg) is surfaced as a
            # tool result the model can self-correct from, never a raw stack trace.
            return {"error": "invalid_argument", "detail": str(exc)}

    def _traversal_refusal(self, question: str) -> AgentResult:
        """Deterministic, zero-LLM-call refusal for trace/path/route/downstream-of/flow-
        simulation questions -- see _TRAVERSAL_BAIT_PATTERN's docstring above. Still fetches
        real 1-hop data for any node id the question mentions (a bounded, cheap, grounded
        lookup -- not a guess) so this doesn't regress the quality of what was previously an
        LLM-authored refusal, just makes it reliable instead of probabilistic."""
        ids = sorted(set(_MENTIONED_ID_PATTERN.findall(question)))
        transcript: list[ToolCallRecord] = []
        lines = [
            "I can't trace multi-hop paths, determine network-wide flow direction, or "
            "simulate what happens if something is closed -- Tier 1 only supports direct "
            "(1-hop) lookups, with no path-finding or flow-simulation tool."
        ]
        if ids:
            lines.append("Here is what I CAN tell you -- each mentioned node's own direct connections:")
            for nid in ids:
                result = self._dispatch("get_neighbors", {"id": nid})
                transcript.append(ToolCallRecord(name="get_neighbors", args={"id": nid}, result=result))
                if isinstance(result, dict) and result.get("error"):
                    lines.append(f"- {nid}: not found on this sheet.")
                    continue
                if not result:
                    lines.append(f"- {nid}: no direct connections recorded.")
                    continue
                for n in result:
                    if n["provenance"] == "TRACED":
                        hedge = "TRACED (a continuously-traced skeleton run)"
                    else:
                        hedge = (
                            "INFERRED -- only a bridged/short-gap connection, not traced as "
                            "continuous ink; verify against the actual drawing before relying on it"
                        )
                    lines.append(f"- {nid} -> {n['neighbor_id']} ({n.get('neighbor_node_type')}): {hedge}")
        return AgentResult(
            text="\n".join(lines), transcript=transcript, grounding=GroundingResult(ok=True),
            rounds_used=0, hit_round_cap=False, refused_by_classifier=True,
        )

    def _judgment_refusal(self, question: str) -> AgentResult:
        """Deterministic, zero-LLM-call decline for subjective/evaluative questions (sheet 8
        K4 audit: the LLM path answered "is this well-instrumented?" with an ungrounded
        verdict -- "modest" -- built on real counts; the grounding checker passed it because
        every FACT was real, it had no notion of answer-type scope. Same structural pattern as
        _traversal_refusal: classify before the LLM ever sees the question, so the decline is
        guaranteed rather than probabilistic. Still does real, grounded retrieval (a node-type
        breakdown, plus the instrument tag listing when the question is about instruments) so
        this isn't a bare non-answer -- it hands over exactly the K1-K3 pattern: decline the
        judgment, offer the real data instead."""
        result = self._dispatch("count_nodes", {})
        transcript: list[ToolCallRecord] = [ToolCallRecord(name="count_nodes", args={}, result=result)]
        breakdown = result.get("breakdown_by_node_type", {}) if isinstance(result, dict) else {}
        lines = [
            "I can't render a subjective or qualitative verdict about this sheet -- no tool in "
            "this system computes a design-quality, adequacy, or risk assessment, so that kind "
            "of conclusion isn't something I can back with graph data, no matter how the real "
            "numbers might look to a human reader. Here is the raw grounded data instead, so "
            "you can form your own judgment:",
        ]
        if breakdown:
            for nt, n in sorted(breakdown.items()):
                lines.append(f"- {nt}: {n}")
            lines.append(f"- total nodes: {result.get('count')}")
        if "instrument" in question.lower():
            listing = self._dispatch("list_systems_or_lines", {})
            transcript.append(ToolCallRecord(name="list_systems_or_lines", args={}, result=listing))
            tags = listing.get("instrument_tags", []) if isinstance(listing, dict) else []
            if tags:
                lines.append("Instrument tags present on this sheet:")
                for t in tags:
                    lines.append(f"- {t['id']}: {t.get('function')}-{t.get('loop_number')} ({t.get('raw_text')})")
        return AgentResult(
            text="\n".join(lines), transcript=transcript, grounding=GroundingResult(ok=True),
            rounds_used=0, hit_round_cap=False, refused_by_classifier=True,
        )

    def answer(self, question: str) -> AgentResult:
        if _is_traversal_bait(question):
            return self._traversal_refusal(question)
        if is_judgment_bait(question):
            return self._judgment_refusal(question)

        messages: list[Message] = [Message(role="user", content=question)]
        transcript: list[ToolCallRecord] = []
        expected = expected_filters(question, self.gi)
        required = required_filters(expected) if expected else {}
        already_nudged = False
        grounding_retries = 0
        redundant_blocks = 0

        for round_i in range(self.max_rounds):
            try:
                turn = self.provider.step(SYSTEM_PROMPT, messages, self.tool_schemas)
            except ProviderError as exc:
                return AgentResult(
                    text="", transcript=transcript, grounding=GroundingResult(ok=False),
                    rounds_used=round_i, hit_round_cap=False, error=str(exc),
                )

            if turn.kind == "final_answer":
                text = turn.text or ""
                grounding = check_grounding(text, transcript, question_text=question, gi=self.gi)
                if grounding.ok or grounding_retries >= MAX_GROUNDING_RETRIES:
                    # sec 3.1/2.3 extended: a filter-completeness violation is a REJECTED
                    # answer just like an uncited id, not merely flagged in the report -- if
                    # retries are exhausted it still returns, but grounding.ok says why.
                    return AgentResult(
                        text=text, transcript=transcript, grounding=grounding,
                        rounds_used=round_i + 1, hit_round_cap=False,
                        grounding_retries=grounding_retries,
                        redundant_tool_calls_blocked=redundant_blocks,
                    )
                # Self-correction, not just a report: push the violation back as a message
                # and give the model another turn instead of returning an answer we already
                # know is invalid. Bounded by MAX_GROUNDING_RETRIES so this can't loop forever.
                grounding_retries += 1
                messages.append(Message(role="assistant", content=text))
                messages.append(Message(
                    role="user",
                    content=(
                        "That answer is not sufficiently grounded and cannot be accepted: "
                        + "; ".join(grounding.violations)
                        + ". Call the specific tool combination described above, then answer again."
                    ),
                ))
                continue

            tc = turn.tool_call

            if already_nudged:
                # C4 termination guard (sheet 8 audit, docs/phase6_tier1_design/
                # stress_results.md): the question's full filter combination was already
                # satisfied by an earlier call this turn -- the soft nudge below told the
                # model so, but a further tool call here is exactly the "re-verify each
                # returned id one by one with get_node" pattern that previously burned the
                # whole round budget without ever reaching a final_answer attempt. Block the
                # dispatch outright (no real tool work happens, nothing new enters the
                # transcript) and force a hard stop instead of a second soft reminder --
                # this is what turns "should conclude" into "cannot avoid concluding."
                redundant_blocks += 1
                result = {
                    "error": "redundant_call_blocked",
                    "detail": (
                        "You already have a complete, grounded answer to this question from "
                        "an earlier tool call this turn. Do not call any more tools -- answer "
                        "the question now using the results you already have."
                    ),
                }
                messages.append(Message(role="assistant", tool_call=tc))
                messages.append(Message(
                    role="tool", name=tc.name, tool_call_id=tc.call_id,
                    content=json.dumps(result, default=str),
                ))
                continue

            result = self._dispatch(tc.name, tc.args)
            transcript.append(ToolCallRecord(name=tc.name, args=tc.args, result=result))
            messages.append(Message(role="assistant", tool_call=tc))
            tool_message_content = json.dumps(result, default=str)
            messages.append(Message(
                role="tool", name=tc.name, tool_call_id=tc.call_id,
                content=tool_message_content,
            ))

            # Cross-sheet audit (C1, docs/phase6_tier1_design/stress_results.md): the model
            # sometimes gets a fully-specified, conclusive result and keeps probing anyway
            # until the round budget runs out. A one-time, contextually-timed nudge right
            # after the call that actually satisfies every filter the question specifies --
            # not a static prompt paragraph the model has to remember six rounds later. Any
            # tool call after this point is blocked outright (above), not just discouraged.
            if not already_nudged and required and _call_satisfies(transcript[-1], required):
                already_nudged = True
                messages.append(Message(
                    role="user",
                    content=(
                        "Note: that call already combined every filter this question "
                        "specifies. Its result is your complete, definitive answer -- "
                        "you do not need further tool calls to answer this question. Any "
                        "further tool call will be blocked; answer now."
                    ),
                ))

        # sec 5: round budget exhausted -> explicit non-answer, never the model's last
        # unfinished train of thought presented as if it were the answer.
        return AgentResult(
            text="I couldn't produce a grounded answer within the tool-call budget.",
            transcript=transcript, grounding=GroundingResult(ok=True),
            rounds_used=self.max_rounds, hit_round_cap=True, grounding_retries=grounding_retries,
            redundant_tool_calls_blocked=redundant_blocks,
        )
