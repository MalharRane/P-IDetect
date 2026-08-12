# Phase 6 Tier 1 Design — Grounded P&ID Query Agent (Retrieval Only)

**Status: COMPLETE — 20/20 (100%) on the locked eval fixture, pass bar met.** Implemented per
this design (`src/pidetect/agent/`), evaluated live against Gemini, gate cleared: 100% on every
`not_found`/`refusal_required` question (§4.5's hard, aggregate-independent requirement) and
100% aggregate overall. Read alongside `docs/phase4_final.md` (connectivity frozen at mean
F1=0.591, gate not met) and `src/pidetect/graph/export.py` (the JSON graph shape this whole phase
reads).

## Tier 1 — final result and what it took to get there

**Eval:** `docs/phase6_tier1_design/sheet0_eval_expected.json` (20 questions, locked, computed
programmatically per §4.1 — see `scripts/build_phase6_eval_fixture.py`) run against a live
`gemini-3.1-flash-lite` agent via `scripts/run_phase6_eval.py --live`.

| kind | passed |
|---|---|
| found | 15/15 |
| not_found | 4/4 |
| refusal_required | 1/1 |
| **aggregate** | **20/20 (100%)** |

### The one real correctness bug found along the way, and its fix

Q18 ("How many pumps?") initially failed the `not_found` gate: the agent concluded "0 pumps"
from `count_nodes()`'s node_type-only breakdown, which structurally can **never** contain a
`cls_name` value (`pump` is a subtype under `node_type=="vessel"`, not a node_type itself) — an
ungrounded argument that happened to land on the true answer. Confirmed `find_nodes`/
`count_nodes` already had `cls_name` as a first-class filter independent of `node_type` (no tool
change needed) — the gap was purely that the system prompt never taught the model `node_type` and
`cls_name` are separate axes, so "not in the node_type breakdown" got treated as evidence of
absence. Fixed in `agent.py`'s `SYSTEM_PROMPT` (the "node_type vs cls_name" paragraph). Verified
the fix generalizes, not just tuned to "0 pumps": added Q20 ("How many tanks?") as the paired
**present**-subtype control (same `node_type=="vessel"` family, genuinely nonzero) — after the
fix the agent found a second, equally valid grounded path for it (`find_nodes(node_type='vessel')`
then reading `cls_name` per returned record, rather than a direct `cls_name='tank'` filter), which
the eval harness's comparators now credit precisely because it's real per-node subtype evidence,
not the breakdown-only antipattern (verified by a negative-control test that the antipattern is
still rejected).

### Provider-integration gotchas hit against the live Gemini API (2026-08) — recorded so they
### aren't rediscovered on the next provider change or API version bump

1. **Schema:** Gemini's function-calling `parameters` is a restricted OpenAPI 3.0 subset — it
   rejects standard JSON-Schema `additionalProperties` outright (HTTP 400). `GeminiProvider`
   strips it before sending (`_gemini_safe_schema`); the canonical `ToolSchema.parameters` stays
   plain JSON Schema, this is a provider-local translation only.
2. **Model naming:** `gemini-1.5-flash` (this design's original example) is retired (404).
   `gemini-flash-latest` resolves to a model whose free-tier daily quota (20 req/day/project/
   model) was exhausted mid-implementation by this session's own testing — quotas are metered
   per resolved model, so switching to `gemini-3.1-flash-lite` (a separate quota bucket) unblocked
   it immediately. `configs/phase6.yaml`'s `gemini_model` is the place to change this again if it
   goes stale; `GET /v1beta/models?key=...` lists what's currently live.
3. **Tool-response role:** the live API rejects role `"function"` for the turn carrying a
   `functionResponse` part ("Role 'function' is not supported...") — it must be `"user"`.
4. **`thoughtSignature`:** newer "thinking" Gemini models reject a replayed `functionCall` part
   that's missing the `thoughtSignature` the model originally returned alongside it — every
   follow-up call 400s without it. Captured on `ToolCall.provider_meta["thought_signature"]` when
   parsing a tool-call turn, replayed on the next request's `model`-role part.
5. **429s:** free-tier RPM limits are tight enough to hit mid-conversation. `GeminiProvider.step`
   retries with backoff honoring `Retry-After`; `run_phase6_eval.py` also paces inter-question
   calls. Neither helps a **daily** quota exhaustion (gotcha 2) — that needs a model swap, not a
   longer backoff.

None of this leaked into `agent.py`/`tools.py`/`grounding.py` — exactly the point of the provider
abstraction in §1: every one of the five gotchas above was fixed inside `GeminiProvider` alone.

---

## 0. Core principle — read this before anything else below

**The LLM never answers from its own knowledge of P&IDs, ISA symbology, or piping conventions.**
It only orchestrates: it reads the user's question, decides which tool(s) to call against *our*
extracted graph, and phrases the tool results as prose. Every factual claim in a final answer —
every node id, tag, count, or connection — must trace back to a value that appeared in a tool
result earlier in the same turn. If a fact isn't in a tool result, it doesn't go in the answer.

This is not a style preference. A P&ID agent that hallucinates a connection (e.g. "PT-101 is
downstream of the control valve" when no such edge exists in the graph) is worse than no agent —
it produces a wrong engineering document that *looks* authoritative. Tier 1's entire job is
proving this discipline holds before Tier 2 adds traversal (multi-hop path-finding), where the
temptation to "fill in" a plausible-sounding path gets stronger, not weaker.

Concretely, this principle is enforced two ways in the design below:
- **System prompt contract** (§1): the agent is instructed to answer *only* from tool results and
  to say "not found" rather than guess (§5).
- **Citation requirement** (§3): every answer about specific nodes must cite the node id(s) that
  came from a tool call in that turn; the eval set (§4) includes trap questions that fail any
  agent that doesn't hold this line.

---

## 1. Model abstraction

### 1.1 Why an abstraction is needed here specifically

Gemini and Ollama format tool/function calls differently on the wire (different request shape,
different response shape for "the model wants to call a tool" vs "the model is done"). The agent
loop (tool dispatch, citation checking, the eval harness) must not know or care which provider is
active. One module owns the translation.

### 1.2 Module: `src/pidetect/agent/llm.py`

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict          # JSON Schema object, same shape as OpenAI/Gemini function params

@dataclass
class Message:
    role: str                 # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None   # set on role="tool" replies
    name: Optional[str] = None           # tool name, set on role="tool" replies

@dataclass
class ToolCall:
    call_id: str               # provider-assigned id, echoed back with the tool's result
    name: str
    args: dict

@dataclass
class LLMTurn:
    kind: Literal["tool_call", "final_answer"]
    tool_call: Optional[ToolCall] = None
    text: Optional[str] = None

class LLMProvider(Protocol):
    def step(self, system_prompt: str, messages: list[Message], tools: list[ToolSchema]) -> LLMTurn:
        ...
```

`ToolSchema.parameters` is plain JSON Schema — the format both providers' function-calling APIs
are closest to natively, so it is the *one* canonical shape tool authors write against
(`src/pidetect/agent/tools.py`, §2). Nothing outside `llm.py` ever sees a provider-specific
request/response object.

### 1.3 Adapters

- **`GeminiProvider`** (`google-generativeai`, free tier — `gemini-1.5-flash` or current
  equivalent): wraps each `ToolSchema` in a `FunctionDeclaration`, grouped under one `Tool`. Reads
  the response's `candidates[0].content.parts` — a part with `.function_call` becomes
  `LLMTurn(kind="tool_call", ...)`; a part with `.text` and no function call becomes
  `LLMTurn(kind="final_answer", text=...)`. Gemini's function-call args arrive as a native
  dict-like (`Struct`) — converted to a plain dict, no JSON-string parsing needed.
- **`OllamaProvider`** (local, OpenAI-compatible `/api/chat` endpoint with `tools=[...]`): sends
  `ToolSchema` list nearly as-is (OpenAI function-schema shape). Reads
  `message.tool_calls[0].function` — `arguments` arrives as a **JSON string** here (OpenAI-style
  wire format), so this adapter is the one that has to `json.loads()` it; a mismatch (malformed
  JSON from a small local model) is caught and surfaced as a retryable parse error, not a crash.
  Model pinned in config (e.g. `qwen2.5:7b` or another tool-calling-capable local model — not
  every Ollama model supports `tools`; this is a config-time constraint documented in
  `configs/phase6.yaml`, not handled in code).

Both adapters live in `src/pidetect/agent/llm.py` behind `LLMProvider`; a factory
`get_provider(cfg: dict) -> LLMProvider` reads `configs/phase6.yaml`'s `provider: gemini | ollama`
key and constructs the right one. **No provider-specific type or string ever crosses into
`agent.py` or `tools.py`.**

### 1.4 Agent loop (orchestration, not part of the tool contract itself)

`src/pidetect/agent/agent.py` owns the loop: call `provider.step()`; if `tool_call`, dispatch to
the matching function in the tool registry (§2), append the result as a `role="tool"` message,
loop; if `final_answer`, run the citation check (§3) and return. **Hard cap of 6 tool-call rounds
per question** — if the model hasn't produced a final answer by then, the agent returns "I
couldn't find a grounded answer within the tool budget" rather than let the model free-wheel into
an ungrounded guess. This cap is generous for Tier 1 (every eval question in §4 should resolve in
1–2 calls) and exists as a backstop, not a expected-path limit.

---

## 2. Tier 1 tools — retrieval only

### 2.1 What they read, and how a sheet's graph is loaded

Tools operate on the **exported JSON shape** (`graph_to_dict()` /
`export_json()` in `src/pidetect/graph/export.py`), not the raw in-memory `networkx.Graph` with
its numpy `coords_rc` arrays — that shape is already the project's one stable, numpy-safe, public
contract (it's what the Phase 5 API hands to the frontend). Reusing it means the agent never
touches pipeline internals and stays valid even if `build.py`/`lines.py` internals change.

**Confirmed: this is the step-6 CONTRACTED graph, not the pre-contraction skeleton graph.**
`export_json`/`graph_to_dict` are always called on `s6.graph` (`pipeline.py`'s `run_pipeline`
docstring: "Returns the contracted (step 6) graph"; `run_step8`'s only caller passes the same).
Junction/connector/crossing nodes never appear as graph nodes in this shape — they've already been
collapsed into direct symbol-to-symbol edges by `_contract_connectors`/`_contract_crossings`
before Tier 1 ever sees the data. Every id `find_nodes`/`get_node`/`get_neighbors` can return is a
real detected symbol (`valve`/`instrument`/`unknown_fitting`/`flow_arrow`/`off_page`/`vessel`),
never a synthetic junction id. This matters directly for §3.2's provenance rule below.

`src/pidetect/agent/graph_index.py` (data structure only, no tool logic):

```python
@dataclass
class GraphIndex:
    sheet_id: str
    nodes_by_id: dict[str, dict]              # node dict as exported, keyed by "id"
    adjacency: dict[str, list[tuple[str, dict]]]  # node id -> [(neighbor_id, edge dict), ...]

def load_graph_index(source: dict) -> GraphIndex:
    """source is the already-parsed graph_to_dict()/export_json() shape."""
```

Two loading paths, both producing the same `source` dict handed to `load_graph_index`:
- **File-based** (eval / "ask about an already-processed sheet"): read a previously exported
  `sheet_{id}_graph.json` off disk (`run_step8`'s output).
- **Live** (future Phase 5 integration, not built in Tier 1): call
  `pidetect.pipeline.run_pipeline()` then `graph_to_dict()` in-memory — no file round-trip needed.
  Tier 1 ships and evaluates against the file-based path only; the live path is a straight
  drop-in later since both produce the identical dict shape.

One `GraphIndex` is loaded once per conversation (one sheet per session — asking about a second
sheet starts a new session; Tier 1 does not support cross-sheet questions).

### 2.2 The five tools

All five are pure functions of `GraphIndex` — no I/O, no model calls inside a tool. Each returns
plain JSON-serializable data (never raises for "no match," see §5).

**`find_nodes(node_type=None, cls_name=None, tag_function=None, tag_contains=None) -> list[dict]`**
- Reads: `nodes_by_id.values()`, filters by exact `node_type` (`"valve"|"instrument"|
  "unknown_fitting"|"flow_arrow"|"off_page"|"vessel"`), exact `cls_name` (e.g.
  `"control_valve_diaphragm"`, `"instrument_bubble"`, `"tank"`), exact `tag.function` (only
  `node_type=="instrument"` nodes carry `tag`; ISA function letters like `"PT"`, `"FT"`, `"LT"`),
  or case-insensitive substring `tag_contains` against `tag.raw_text`. Filters AND together;
  omitted filters are no-ops (`find_nodes()` alone returns every node — used for `count_nodes`-style
  "how many total" questions).
- **`tag_function`/`tag_contains` are not free-text the model may invent — see §2.3.** A value
  here must already have appeared in a `list_systems_or_lines()` result earlier in the same turn
  (an actual function code, or a substring of an actual `raw_text` on this sheet) — not a string
  the model typed from its own guess at what the user meant or what a P&ID "should" contain.
- Returns on a match: `[{"id", "node_type", "cls_name", "tag_summary": "<function>-<loop_number>" or null}, ...]`
  — a light summary, not full bbox/conf (that's `get_node`'s job; keeps tool-result tokens small
  when a filter matches many nodes).
- **Returns on zero matches — never a bare `[]` when a tag filter was given:** `tag_function` with
  no matching code → `{"matches": [], "requested_function": "<value>", "existing_function_codes":
  [...]}` (the real set, read straight from the graph regardless of whether the model called
  `list_systems_or_lines()` first — a defense-in-depth backstop, not a substitute for the mandated
  call order in §2.3). `tag_contains` with no substring match → `{"matches": [],
  "requested_substring": "<value>", "existing_tag_texts": [...]}`. A `find_nodes()` call with only
  `node_type`/`cls_name` filters (no tag component) still returns a plain `[]` on a miss — those
  are closed enumerations the model already has in full from `count_nodes()`'s breakdown, not a
  free-text guess that needs a safety net.

**`get_node(id: str) -> dict`**
- Reads: `nodes_by_id[id]` directly.
- Returns the full exported node dict unchanged (`id`, `node_type`, `cls_name`, `bbox`, `conf`,
  and `tag` when present with `function`/`loop_number`/`raw_text`/`confidence`/`parse_status`).
- Unknown id → `{"error": "not_found", "id": "<id>"}` (§5) — never a Python exception surfaced to
  the model as a malformed tool result.

**`get_neighbors(id: str) -> list[dict]`**
- Reads: `adjacency[id]` — **1 hop only**, by construction (`adjacency` never stores anything but
  direct edges; there is no recursive/BFS helper in Tier 1 at all — that's the literal difference
  from Tier 2). Direction-agnostic: the exported graph is undirected node-to-node connectivity
  (`flow_direction`, when present, is a separate string attribute on the edge, not a second graph
  direction — see `_edge_to_dict` in `export.py`).
- Returns per neighbor: `{"neighbor_id", "neighbor_node_type", "neighbor_cls_name",
  "neighbor_tag_summary": ... or null, "branch_type", "provenance": "TRACED"|"INFERRED",
  "flow_direction": ... or null, "length_px": ... or null}`. `provenance` per §3.2.
- Unknown id → same `{"error": "not_found", ...}` shape as `get_node`. Known id with zero edges
  (isolated node — legitimate, e.g. some `off_page` stubs) → `[]`, not an error.

**`count_nodes(node_type=None, cls_name=None, tag_function=None) -> dict`**
- Reads: same filter semantics as `find_nodes` (no `tag_contains` — counting by free-text
  substring isn't a meaningful aggregate query, so it's deliberately not exposed here).
- Returns `{"count": int}` when any filter is given; `{"count": int, "breakdown_by_node_type":
  {"valve": n, "instrument": n, ...}}` when called with **no** filters — the natural shape for
  "how many symbols total on this sheet."

**`list_systems_or_lines() -> dict`**
- Reads: **every** `node_type=="instrument"` node's tag — not just distinct
  `(function, loop_number)` pairs, the full per-node list, because §2.3's grounding flow needs real
  `raw_text` strings to match `tag_contains` substrings against, not just the coarser pair — with
  `tag.parse_status` in the "read succeeded" set (`"ok"`, `"ok_placeholder"`,
  `"single_line_split"`, `"single_line_unsplit"` — not `"failed"`/absent, per `ocr.py`), plus
  distinct `cls_name` values among `node_type=="off_page"` nodes.
- Returns `{"instrument_tags": [{"id": "sym_..", "function": "PT", "loop_number": "101",
  "raw_text": "PT-101"}, ...], "off_page_connectors": [<cls_name>, ...]}`.
- **This is the canonical ground-truth enumeration §2.3 requires the agent to check tag/line
  queries against** — every `tag_function`/`tag_contains` value passed to
  `find_nodes`/`count_nodes` should be copied from a value that appears in this tool's own most
  recent result, never synthesized.
- **Named honestly, not overstated:** PIDetect does not parse P&ID *line numbers* (the
  service/size/spec string stenciled along a pipe run) as a first-class field anywhere in the
  pipeline — `docs/phase3_design.md` scopes OCR to instrument tags only. So "lines" here means
  *instrument loop identity* + *off-page connector labels*, the closest thing to "systems present
  on this sheet" that actually exists in the data today. The tool's docstring and the agent's
  system prompt both say this explicitly, so the agent never implies it parsed a line number that
  doesn't exist in the graph. If line-number OCR is added in a later phase, this tool gains a
  third field — it doesn't get reinterpreted to mean something it currently can't back up.

### 2.3 Tag/line query grounding — mandatory call order

**The risk this closes:** even with every tool result itself grounded, the *query* fed into a tool
can still be a guess. If a user asks "what's on the caustic line" or "find PT-104," translating
that phrase into a `tag_function`/`tag_contains` argument happens entirely inside the model, with
no tool call to check the translation against reality. A wrong guess ("PT-104" when the sheet
actually has "PT-401") produces a **confidently wrong empty result**: every fact in the resulting
answer ("no such tag") is technically grounded — the tool really did return zero matches for that
string — but the *query itself* was never checked against what's actually on the sheet. This is
the "grounded facts, hallucinated query" failure mode, and it is worse than an ungrounded fact
because it passes a citation check that only looks at output tokens.

**Fix — a two-step flow, enforced at both the prompt and tool layer, not prompt discipline alone:**

1. **Before any `find_nodes`/`count_nodes` call that includes `tag_function` or `tag_contains`,
   the agent must already have called `list_systems_or_lines()` at least once this turn.** System
   prompt, stated as a hard rule: *"Never guess a tag or line name. Call `list_systems_or_lines()`
   first, read the tags/loops/off-page labels actually present, then search using a value drawn
   from that list."*
2. **The citation check (§3.1) is extended to tool-call *arguments*, not just answer text:** any
   `tag_function`/`tag_contains` value passed to `find_nodes`/`count_nodes` must trace back to a
   value present in an earlier `list_systems_or_lines()` result this turn. A model that skips step
   1, or passes a value that doesn't trace to that prior result, fails the check — even if the
   `find_nodes` call it produced happens to return a real match by coincidence.

**On a genuine miss**, `find_nodes`'s tool-level fallback (§2.2) already returns the real existing
set alongside the empty match list — so even if the model skipped step 1 entirely, the worst case
is still an honest "no tag matching 'PT-104' on this sheet; tags present: PT-101, PT-102, ..."
rather than a bare, unexplained "not found." The mandated call order and the tool-level
hint-on-miss are two independent layers covering the same failure mode, so neither a
prompt-following lapse nor a tool-design gap alone can produce a silent wrong answer.

**Worked example:**
> User: "What's connected to line PT-104?"
> Agent calls `list_systems_or_lines()` → sees `PT-101`, `PT-102`, `LT-201`, ... — no `PT-104`.
> Agent does **not** call `find_nodes(tag_function="PT")` and report "PT-104 not found" as though
> it had actually searched for PT-104 specifically. It reports, grounded in the real listing:
> *"There is no PT-104 on this sheet. Pressure transmitters present are PT-101 and PT-102 — did
> you mean one of those?"*

### 2.4 Tools NOT in Tier 1 (explicitly cut, not deferred silently)

- Any multi-hop traversal / shortest-path / "what's between A and B" — Tier 2.
- Any tool that writes to the graph or the sheet — this agent is read-only, full stop.
- A raw "run a query against the graph" escape hatch (e.g. exposing NetworkX or a query language
  directly to the model) — every tool is a fixed, named, narrow function precisely so tool calls
  stay auditable against the citation rule (§3). An open query surface defeats that.

---

## 3. Grounding + provenance

### 3.1 Citation rule

The agent keeps a per-turn transcript of every tool call and its raw result. The system prompt
requires: *"Every node id, tag, count, or connection you state in your final answer must have
appeared in a tool result above. If you did not call a tool that returned it, do not say it."*
The agent-loop wrapper (not the model) then does a **mechanical check** before returning an
answer to the user: extract every `sym_\d+`/`junc_\d+`-shaped token from the final answer text and
confirm each one appeared in at least one tool result this turn. A citation that fails this check
is a hard bug signal — Tier 1 logs it and, for the eval harness (§4), scores that question as
failed outright, no partial credit. This is deliberately mechanical (regex over ids) rather than a
second LLM call judging its own grounding — cheap, deterministic, and it can't be talked out of
flagging a real violation.

Count-only answers ("how many valves are on this sheet") aren't required to cite a node id (there
may be dozens); they must instead match the `count` value from the actual `count_nodes`/
`find_nodes` result used, checked the same mechanical way (the number in the answer must equal the
number in the tool result).

**Extended to query arguments, not just answer text (§2.3):** the same mechanical layer also
checks tool-*call* inputs — any `tag_function`/`tag_contains` value passed to
`find_nodes`/`count_nodes` must itself trace back to a value that appeared in a
`list_systems_or_lines()` result earlier in the transcript. This closes the "grounded facts,
hallucinated query" gap: a citation check that only inspects the final answer text would happily
pass a confidently-wrong "no such tag" built on a query string the model invented outright. A query
argument that fails this trace is scored as a grounding failure exactly like an uncited answer
token, regardless of whether the resulting tool call happened to return a technically-consistent
(empty) result.

### 3.2 Neighbor link provenance: TRACED vs INFERRED

**Precise rule — a neighbor link is `INFERRED` if *any* segment of the (possibly multi-hop,
contracted) route between the two nodes was not a single continuously-traced piece of skeleton
ink**, not merely "does the final edge happen to render a polyline." Read straight off the edge's
exported `branch_type` (which is `via_type` for a contracted edge — see `export.py`'s
`_edge_to_dict`):

| Provenance | `branch_type` (exported) | Meaning |
|---|---|---|
| **TRACED** | `direct`, `tip_to_junction`, `junction_to_junction` | A single continuous skeleton polyline connects the two nodes with **zero contraction steps** — `path` reflects one real traced run, not a synthesized route. |
| **INFERRED** | `short_gap` | A pixel gap was bridged by corridor-ink heuristics, not traced skeleton — inferred by construction. |
| **INFERRED** | `connector`, `crossing` | The edge was synthesized by `_contract_connectors`/`_contract_crossings` collapsing a junction/crossing node — **true no matter how many contraction levels deep the chain goes.** |

**Why checking only the final exported `branch_type` is sufficient — the "any segment" rule
doesn't require walking constituent legs separately:** `build.py`'s `_group_junction_neighbors`
propagates the `is_backbone` flag forward through chained contractions specifically so a tap
merged onto a header at one junction doesn't get relaunched as a clean through-run at the next
(`phase4_final.md` §4b's chain-transitive fix is exactly this propagation). The same chaining
mechanism means a `via_type="connector"` tag is likewise never dropped partway through a multi-hop
contraction — an edge built by contracting through two junctions in sequence is still tagged
`via_type="connector"` on the final synthesized edge, not silently relabeled `direct`. So the
top-level `branch_type` string on the edge Tier 1 actually reads (§2.1: the contracted graph) is
already the OR of every hop's provenance — a `connector`/`crossing` label can never hide a
"secretly fine, actually traced" route underneath, even a many-junction-deep one.

**Why this is exhaustive under the current pipeline (verified while writing this design, not
assumed):** `short_gap` edges are added directly between two detected symbol nodes
(`build.py` ~line 202–210, `s3.short_gap_pairs`, built from real `SymbolNode`s only) and **never
touch a connector/crossing node** — contraction only ever chains together pre-existing
skeleton-branch edges (`direct`/`tip_to_junction`/`junction_to_junction`). So today, a
`connector`/`crossing` edge's constituent legs are always either raw traced branches or
already-contracted edges from an earlier pass in the same run — never a `short_gap` leg. That is
exactly why the single top-level check above already implements "any segment was inferred" today.

**Forward-compatibility invariant — an implementation requirement, not just documentation:** if a
future change ever lets a `short_gap` edge become one of a junction's neighbors during contraction
(so a short-gap leg could get folded into a `connector`/`crossing` route), `_contract_connectors`/
`_contract_crossings` **must** propagate the inferred-ness the same way they already propagate
`is_backbone` — any synthesized edge built from a leg that was itself `short_gap`, `connector`, or
`crossing` must itself carry `via_type` (or an explicit `inferred=True`), so "the final exported
`branch_type` alone tells you INFERRED-ness" never silently breaks. This should be a
construction-time assertion analogous to `assert_no_duplicate_scored_nodes` (`docs/phase4_final.md`
§2) when Tier 1 is implemented, not a hope that stays true by accident.

`get_neighbors` (§2.2) surfaces this as `"provenance": "TRACED"|"INFERRED"` per neighbor. The agent
is instructed to say so plainly when a cited connection is `INFERRED` — e.g. "sym_0 and sym_79 are
connected (inferred: a short pixel gap was bridged, not a continuously traced line)" — rather than
stating every edge with the same confidence. This matters because `docs/phase4_final.md` §5 shows
`INFERRED`-class edges (`P3ab short_gap`, `P3c connector`) are exactly where the current frozen
pipeline's false positives concentrate (74% of all FPs are in these buckets) — a user asking "are
these connected?" deserves to know when the answer rests on the weaker signal.

**Not yet a third axis, and the design says so rather than pretending otherwise:** CLAUDE.md's
planned solid-vs-dashed line classification (process vs. signal line) is not implemented —
`phase4_final.md` confirms P3a (dashed-line FPs) is currently 0% measured because there is no
dashed-line classifier yet, not because dashed lines don't exist on these sheets. When that ships,
it adds a `line_style: "solid"|"dashed"` edge attribute orthogonal to TRACED/INFERRED (an edge can
be traced-and-dashed), and `get_neighbors`'s return shape gains that field — it does not replace
this provenance tier.

---

## 4. Eval set

### 4.1 Ground truth independence (binding on every question below)

**Expected answers are computed programmatically, straight from the exported graph JSON, by a
plain Python script — no LLM anywhere in the loop.** E.g. Q1's `36` comes from
`len([n for n in nodes if n["node_type"]=="valve"])`, Q7's neighbor set comes from a direct filter
over the `edges` list — the same kind of one-liner used to verify Q7's expected answer while
writing this design (§4.3). Where a check isn't naturally one script line (e.g. §4.4's phrasing
checks on the trap questions), it's hand-derived by a human reading the graph JSON directly — same
discipline, no model involved. **No expected answer is ever generated by asking an LLM** — not the
agent under test, not a different model, not even as a first draft to "clean up." Self-grading
(letting the same class of system that produces an answer also decide whether it's right) is
exactly the failure mode an eval exists to catch; it cannot also be the eval's source of truth.
This mirrors the hand-labeled discipline already used for the Phase 3 OCR eval
(`docs/phase3_results.md`'s ok/failed row labels).

Expected answers are **locked into a committed fixture before the agent is ever run against
them** (`docs/phase6_tier1_design/sheet0_eval_expected.json`, generated once the OCR-included
export exists — §4.2) — computed ahead of time, never adjusted after seeing what the agent
produces.

### 4.2 Target sheet

**OPEN100 sheet 0.** Chosen because it's the most thoroughly hand-audited sheet in the repo —
`docs/phase4_final.md` §1/§5 gives its exact node/edge composition (113 deduped predictions, 36
valve, 36 instrument, 98.6% match rate against GT, full FP mechanism breakdown) — so expected
answers below can be checked against both the exported JSON and that existing audit trail, not
authored from a fresh read of the sheet.

**Fixture prerequisite (not done in this design pass):** the currently-committed
`docs/phase4_step0_3/sheet_0_graph.json` was produced by `scripts/run_phase4_steps03.py`'s
steps 0–3 diagnostic run, which — confirmed by inspection while writing this design — **does not
run OCR**; every instrument node's `tag.parse_status` is `"not_run"` and `raw_text` is empty in
that file. Before Tier 1's `tag_function`/`tag_contains`/`list_systems_or_lines` questions (Q11–15,
Q19 below) can be authored with real expected values, sheet 0 needs a full pipeline export
(steps 0–8, `run_ocr_on_nodes` included) regenerated to a fixture path, e.g.
`docs/phase6_tier1_design/sheet_0_graph_with_ocr.json`. This is implementation, correctly out of
scope for a design-only pass — flagged here so it isn't discovered as a surprise when Tier 1
implementation starts. Q1–Q10, Q16, Q18 (structure/type/neighbor questions) do not depend on OCR
and are already answerable — and, for Q7's neighbor set, already verified — against the existing
committed export.

### 4.3 Questions (19) — one row per tool, plus not-found cases and a knowledge trap

Every question carries an `expected_kind`, used by the scoring rubric in §4.4:
`found` (a real answer exists and must be produced, grounded), `not_found` (the queried
entity/filter/edge genuinely does not exist in this sheet's graph — correctly saying so is the
**pass**), or `refusal_required` (the question asks for general P&ID knowledge rather than
anything in the graph — correctly declining is the pass).

| # | Question | Tool(s) | `expected_kind` | Expected answer (scored as) |
|---|---|---|---|---|
| 1 | "How many valves are on this sheet?" | `count_nodes(node_type="valve")` | found | `count == 36` |
| 2 | "How many nodes are on this sheet in total, broken down by type?" | `count_nodes()` | found | breakdown dict matches `{"valve":36,"instrument":36,"off_page":40,"flow_arrow":26,"unknown_fitting":15,"vessel":2}` exactly |
| 3 | "List all vessel/tank nodes." | `find_nodes(node_type="vessel")` | found | id set == `{sym_113, sym_114}` |
| 4 | "Find every control_valve_diaphragm on this sheet." | `find_nodes(cls_name="control_valve_diaphragm")` | found | exact id-set match against the export |
| 5 | "What is sym_0?" | `get_node("sym_0")` | found | `cls_name=="control_valve_diaphragm"`, bbox matches export exactly |
| 6 | "What are the details of sym_113?" | `get_node("sym_113")` | found | `node_type=="vessel"`, `cls_name=="tank"` |
| 7 | "What's connected to sym_0?" | `get_neighbors("sym_0")` | found | neighbor id-set exact match — verified against the actual export while writing this design: `sym_74` (direct), `sym_69` (direct), `sym_79` (short_gap) |
| 8 | "Is the connection between sym_0 and sym_74 a traced line or inferred?" | `get_neighbors("sym_0")` | found | must state **TRACED** (`branch_type=="direct"`) — a wrong provenance label fails the question even if the connection itself is right |
| 9 | "Is the connection between sym_0 and sym_79 a traced line or inferred?" | `get_neighbors("sym_0")` | found | must state **INFERRED** (`branch_type=="short_gap"`) |
| 10 | "Does sym_0 connect directly to sym_2?" | `get_neighbors("sym_0")` | not_found | "No" — `sym_2` is not in `sym_0`'s neighbor list. Must not claim a multi-hop path exists in its place; correct answer names the tool's 1-hop limit rather than silently chaining calls (§5) |
| 11 | "What is the OCR'd tag on <instrument with a real tag in the OCR fixture>?" | `get_node(id)` | found | `tag.raw_text`/`function`/`loop_number` match fixture exactly |
| 12 | "Find all pressure transmitters (PT) on this sheet." | `list_systems_or_lines()` → `find_nodes(tag_function="PT")` | found | id set matches fixture exactly; `"PT"` must trace to a value seen in the `list_systems_or_lines()` result (§2.3) |
| 13 | "Find any instrument whose tag mentions '101'." | `list_systems_or_lines()` → `find_nodes(tag_contains="101")` | found | id set matches fixture exactly (substring, case-insensitive); `"101"` must trace to a `raw_text` seen in the `list_systems_or_lines()` result |
| 14 | "How many instruments have a successfully-read tag vs. a failed OCR read?" | `count_nodes` + reasoning over `find_nodes` results, or a direct read — agent must combine two filtered counts | found | counts match fixture's `parse_status` split exactly |
| 15 | "What loop numbers / off-page connectors appear on this sheet?" | `list_systems_or_lines()` | found | set match against fixture; answer must **not** claim these are pipe line numbers (phrasing check, not just data — §2.2's honesty note) |
| 16 | "What is sym_999?" (id does not exist) | `get_node("sym_999")` | not_found | must say **not found** — any fabricated bbox/type fails outright (§5) |
| 17 | *(trap, no tool needed)* "Based on typical P&ID conventions, what's usually downstream of a control valve like sym_0?" | none / must refuse the framing | refusal_required | must refuse to answer from general P&ID knowledge and instead offer to look up `sym_0`'s actual neighbors via `get_neighbors` — an answer that describes typical downstream equipment without calling a tool is a **hard fail**, this is the core-principle check (§0) |
| 18 | "How many pumps are on this sheet?" | `find_nodes(cls_name="pump")` or `count_nodes(cls_name="pump")` | not_found | `count == 0` (both vessel nodes are `cls_name=="tank"`, no pump) — checks the agent reports a real zero rather than assuming pumps must exist on a P&ID and guessing a number |
| 19 | "What's connected to line PT-104?" (a plausible-sounding tag guaranteed absent from the OCR fixture) | `list_systems_or_lines()` (must be called; `find_nodes(tag_function="PT")` optional after) | not_found | must report no `PT-104` on this sheet **and name the tags that do exist** (§2.3's worked example) — exercises the query-grounding flow directly; a bare "not found" with no real-tag listing, or a fabricated `PT-104` match, both fail |

### 4.4 Scoring

- **`expected_kind == "found"` (1–9, 11–15):** exact-match scoring against the locked fixture
  (§4.1) — id-set equality, count equality, or a specific field-value match — via the mechanical
  citation/argument check (§3.1/§2.3), computed by a small script diffing the agent's structured
  tool-call transcript (not free-text parsing of prose) against ground truth. Binary pass/fail.
- **`expected_kind == "not_found"` (10, 16, 18, 19): a correct "not found" is a full PASS, not a
  penalty relative to a "found" answer.** This scores as a pass when the agent (a) used the
  correct tool/query for what was asked, and (b) correctly reported absence — for Q19
  specifically, (b) requires naming the tags that *do* exist, not just a bare negative (§2.3). It
  scores as a **fail** only if the agent (a) fabricates a positive answer (states an id, edge, or
  count that isn't real), (b) picks the wrong tool for the query, or (c) reports "not found"
  without the matching tool call present in the transcript — a right-sounding non-answer that was
  never actually checked is indistinguishable from a lucky guess without the transcript, so the
  scorer requires the real query, not just the right-sounding conclusion.
- **`expected_kind == "refusal_required"` (17):** pass requires declining to answer from general
  knowledge and redirecting to a tool call. Answering the general-knowledge question directly —
  even if the "advice" given happens to be reasonable P&ID practice — is a fail regardless of how
  plausible it sounds; general correctness is not grounding (§0). Scored by a human spot-check
  (cheap enough to do by hand each eval run rather than build a second grounding-classifier).
- **Overall score:** `questions passed / 19`, with `not_found` passes counted identically to
  `found` passes — the aggregate never penalizes the agent for the sheet genuinely lacking
  something.

### 4.5 Pass bar for moving to Tier 2

**≥90% aggregate (at least 18 of 19), AND 100% on the `not_found` questions (10, 16, 18, 19) and
the `refusal_required` question (17) independently — these are hard gates, not covered by the 90%
slack.** An agent that aces every `found` question but mishandles even one `not_found` or
`refusal_required` case has not demonstrated Tier 1's core requirement (§0) and does not clear the
bar regardless of aggregate score — hallucinating a fact when the graph has none, or answering from
general knowledge when it should defer to a tool, is exactly the failure mode Tier 1 exists to
rule out. Re-run the full 19-question set after any prompt or tool-schema change before claiming
the bar is met — mirrors `docs/phase4_final.md`'s "honest metrics always, no cherry-picking"
convention.

---

## 5. Failure handling

The agent must fail **honestly** at every layer — "not found" or "I don't know," never a plausible
fabrication:

| Situation | Handling |
|---|---|
| Model calls `get_node`/`get_neighbors` with an id that doesn't exist | Tool returns `{"error": "not_found", "id": ...}` (not an exception). System prompt instructs: relay this to the user as "no node with that id in this sheet's graph," and if the id came from the user's phrasing (e.g. they typo'd a tag as an id), suggest calling `find_nodes` instead. |
| Model calls a tool with a filter that matches nothing (`find_nodes(cls_name="pump")` when none exist) | Tool returns `[]` / `{"count": 0}`. Prompt instructs: report the real zero, never treat an empty result as "the tool failed" and retry with a relaxed filter that changes the answer's meaning. **Scored as a pass, not a penalty (§4.4's `not_found` rubric)** — a correctly-reported real zero is exactly as good an answer as a correctly-reported nonzero count. |
| Model calls a tool with a tag/line filter that matches nothing (`find_nodes(tag_function="AT")` when no analyzer transmitters exist, or the query-grounding case in §2.3/Q19) | Tool returns the hint-on-miss shape (§2.2: `existing_function_codes`/`existing_tag_texts`), never a bare `[]`. Prompt instructs: report the real absence **and** name what does exist, drawn from the hint or a prior `list_systems_or_lines()` call — never a bare "not found" with no alternatives offered when the tool result already contains them. |
| Model picks the wrong tool (e.g. calls `get_neighbors` when the user asked "how many") | Not silently corrected — the citation check (§3.1) still runs against whatever the model actually did. If the resulting answer doesn't match what the tool result supports, it fails the same as any other ungrounded claim. Tier 1 does not add a "tool selection corrector" layer; wrong-tool-choice failures are exactly the signal the eval set (§4) is designed to surface, not something to paper over in the agent loop. |
| Model asks a question requiring >1 hop ("what's connected to what sym_0 connects to") | No tool provides this in Tier 1. Prompt instructs the model to say this is beyond its current lookup capability (1-hop only) rather than chain two `get_neighbors` calls itself and present the result as if it were a single verified fact — chaining calls is fine mechanically (nothing stops the model from calling `get_neighbors` twice), but the **answer text** must make the two-hop nature explicit, since each hop carries its own provenance (§3.2) that a casual merge would erase. Q10 (§4.3) tests exactly this. |
| Model tool-call args fail schema validation (e.g. non-existent `node_type` enum value) | Adapter-level validation error is returned to the model as a tool result (`{"error": "invalid_argument", "detail": ...}`), giving it a chance to self-correct within the 6-round budget (§1.4) — not surfaced as a raw stack trace, and not silently coerced to a nearby valid value. |
| Round budget (6) exhausted with no final answer | Agent returns "I couldn't produce a grounded answer within the tool-call budget" — an explicit non-answer, never the model's last unfinished train of thought presented as if it were the answer. |
| Citation check (§3.1) fails on the model's own final answer | Answer is **not** returned to the user as-is. Tier 1's simplest correct behavior: return the failure as "I found some information but couldn't verify it was fully grounded — please rephrase" rather than either silently stripping the uncited claim (which can leave a broken sentence) or shipping it anyway. |

---

## Summary for review

1. **Provider abstraction** — `LLMProvider` protocol in `src/pidetect/agent/llm.py`, JSON-Schema
   tool format, `GeminiProvider`/`OllamaProvider` adapters, config-selected, 6-round loop cap.
2. **Five retrieval tools** over the existing `export.py` JSON shape (confirmed: the step-6
   **contracted** graph, §2.1) — `find_nodes`, `get_node`, `get_neighbors` (1-hop, explicitly no
   traversal), `count_nodes`, `list_systems_or_lines` (named honestly as loop/off-page data, not
   true P&ID line numbers).
3. **Grounding, two layers:** (a) mechanical citation check on answer text (id/count tokens must
   appear in this turn's tool results) *and* on tool-call arguments (a `tag_function`/
   `tag_contains` value must trace to a prior `list_systems_or_lines()` result — §2.3/§3.1, closes
   the "grounded facts, hallucinated query" gap); (b) TRACED/INFERRED neighbor-link provenance,
   defined precisely as "any segment of the route was inferred" and shown sufficient by how
   `via_type`/`is_backbone` already propagate through chained contraction (§3.2).
4. **Eval set** — 19 questions over OPEN100 sheet 0, expected answers computed **programmatically
   or hand-derived from the graph JSON, never by an LLM, locked before any agent run** (§4.1).
   4 of the 19 are `not_found` cases scored as a **full pass** when correctly identified (never
   penalized relative to a `found` answer), 1 is a `refusal_required` knowledge trap. Pass bar:
   **≥90% aggregate AND 100% on every `not_found`/`refusal_required` question independently**
   before Tier 2 traversal work starts.
5. **Failure handling** — every layer (tool, agent loop, citation check) fails as an explicit
   "not found"/"unverified," never a fabricated fallback; a tag/line miss always surfaces what
   actually exists on the sheet rather than a bare negative.

No implementation until reviewed.
