"""Phase 6 Tier 1 -- the agent loop (docs/phase6_tier1_design.md sec 1.4).

Owns orchestration only: call provider.step(); dispatch tool_call turns through the
registry (tools.py); run the mechanical grounding check (grounding.py) on the final
answer. No provider-specific code here (llm.py) and no retrieval logic here (tools.py) --
this module just wires the three together plus the round-cap/failure-handling policy
from sec 5.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pidetect.agent.graph_index import GraphIndex
from pidetect.agent.grounding import GroundingResult, ToolCallRecord, check_grounding
from pidetect.agent.llm import LLMProvider, Message, ProviderError, ToolSchema
from pidetect.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MAX_ROUNDS = 6  # sec 1.4: generous backstop, not an expected-path limit

SYSTEM_PROMPT = """You are a grounded P&ID query agent. You answer questions about ONE \
piping-and-instrumentation diagram sheet using ONLY the tools provided -- find_nodes, \
get_node, get_neighbors, count_nodes, list_systems_or_lines.

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
ungrounded argument even when the conclusion happens to be true.

1 HOP ONLY: get_neighbors returns direct connections only. If a question needs more than \
one hop, say that plainly rather than chaining calls and presenting the result as a single \
verified fact.

PROVENANCE: get_neighbors labels each connection TRACED (a single continuously-traced \
skeleton run) or INFERRED (a bridged gap or a junction-contraction reroute -- weaker \
evidence). State this plainly when it's relevant to the question, don't present every \
connection with the same confidence.
"""


@dataclass
class AgentResult:
    text: str
    transcript: list[ToolCallRecord]
    grounding: GroundingResult
    rounds_used: int
    hit_round_cap: bool = False
    error: str | None = None


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

    def answer(self, question: str) -> AgentResult:
        messages: list[Message] = [Message(role="user", content=question)]
        transcript: list[ToolCallRecord] = []

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
                grounding = check_grounding(text, transcript, question_text=question)
                return AgentResult(
                    text=text, transcript=transcript, grounding=grounding,
                    rounds_used=round_i + 1, hit_round_cap=False,
                )

            tc = turn.tool_call
            result = self._dispatch(tc.name, tc.args)
            transcript.append(ToolCallRecord(name=tc.name, args=tc.args, result=result))
            messages.append(Message(role="assistant", tool_call=tc))
            messages.append(Message(
                role="tool", name=tc.name, tool_call_id=tc.call_id,
                content=json.dumps(result, default=str),
            ))

        # sec 5: round budget exhausted -> explicit non-answer, never the model's last
        # unfinished train of thought presented as if it were the answer.
        return AgentResult(
            text="I couldn't produce a grounded answer within the tool-call budget.",
            transcript=transcript, grounding=GroundingResult(ok=True),
            rounds_used=self.max_rounds, hit_round_cap=True,
        )
