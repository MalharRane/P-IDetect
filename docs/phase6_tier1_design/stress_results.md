# Phase 6 Tier 1 — Stress Test Results

**Source questions:** `docs/phase6_tier1_design/sheet0_stress_questions.json` (28 questions, 7
categories, ground truth computed programmatically from a real graph — see that file's own
`build_phase6_stress_questions.py` for how).

**Run manually** by the user through the chat UI (`ChatPanel.jsx` / `/jobs/{job_id}/query`)
against a **live-uploaded** sheet-0 job (`job_id = 83271c423b6340aa95e19e171d8d931f`), not the
static `sheet0-demo` fixture the questions were originally authored against. The live upload
re-runs detection from scratch, so node ids differ from the locked fixture (`sym_0`'s real
neighbors here are `sym_71`/`sym_76`/`sym_81`, not the fixture's `sym_69`/`sym_74`/`sym_79`) —
ground truth below was recomputed against *this* job's actual graph
(`docs/phase4_step0_3/live_upload_83271c_graph.json`) before grading, not the original fixture
numbers. One confirmed, consequential difference between the two: **the live pipeline never
creates vessel/tank nodes** (0 here vs. 2 in the fixture, which injects them from ground truth
via a diagnostic-only script) — a real production limitation, not an agent defect.

Where the pasted transcript didn't include an explicit tool-call list, that's marked
**not captured** below rather than reconstructed — nothing in this file is inferred beyond what
was actually pasted or what the tool-provenance data in the pasted transcript directly implied
(marked **inferred** in those cases).

## Summary

| | count |
|---|---|
| PASS | 19 |
| FAIL | 6 |
| UNCLEAR (ambiguous transcript, not graded) | 2 |
| NOT RUN | 1 |
| **Total** | **28** |

**Top findings:**
1. **Round-cap exhaustion is systemic** — hit on 4 of 24 graded questions (M1, M2, M3, O2), always
   on compound/multi-step retrieval. M2 needs 13 tool calls (1 `find_nodes` + 12 `get_neighbors`)
   against a 6-round budget — structurally unfinishable regardless of strategy.
2. **O3 is a confirmed core-principle violation the grounding check misses** — chained two 1-hop
   `get_neighbors` calls and presented the result as a validated "connected through" relationship,
   exactly what the system prompt forbids. Passed the mechanical grounding check anyway (every id
   used was individually real) — a confirmed blind spot: the check verifies id-presence, not
   whether a synthesized relationship between two real ids is warranted.
3. **The live upload pipeline never creates vessel/tank nodes** — a real production gap (see
   above), not an agent misbehavior. Caused an initial mis-grading of A3 as a fabrication before
   it was traced down to this.
4. **Knowledge-trap refusal is fully robust** — 4/4 (K1–K4) cleanly refused general P&ID/
   engineering knowledge with no inconsistency.

---

## 1. Ambiguity

*Vague referents, undefined orderings, no unique answer.*

### A1 — "Show me the valve."
- **Expected:** clarify — 36 valves match, no unique referent.
- **Pasted answer:** not tested.
- **Tool calls:** —
- **Problems:** —
- **Verdict: NOT RUN**

### A2 — "What's the second valve on the sheet?"
- **Expected:** clarify — no ordering field exists anywhere in the schema.
- **Pasted answer:** "The second valve identified on the sheet is node `sym_1`, which is a
  `control_valve_diaphragm`."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** Silently resolved an undefined ordinal ("second") and asserted one answer as
  fact, with no ordering ever defined by any tool.
- **Verdict: FAIL**

### A3 — "What's the pressure transmitter near the tank?"
- **Expected:** clarify — 4 PTs exist, no spatial-proximity tool.
- **Pasted answer:** "The available instrumentation on this sheet includes the following pressure
  transmitters (PT): PT 14088 (sym_91), PT 14086 (sym_92), PT 14085 (sym_95), and PT 14087
  (sym_96). However, there are no vessels identified on this sheet, so I cannot determine which
  transmitter might be located near one."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** Initially mis-graded as fabricated — checked against the wrong job at the time.
  Re-verified against `job=83271c...`'s real graph: every id/tag pairing is correct, and "no
  vessels identified" is genuinely true for this job. Correctly declined to guess proximity.
- **Verdict: PASS**

### A4 — "Which instrument is the most critical one?"
- **Expected:** refuse — no "criticality" field exists.
- **Pasted answer:** "The provided data does not contain information designating any specific
  instrument as 'most critical.' The P&ID documentation lists various instruments—including
  Temperature Elements (TE), Pressure Transmitters (PT), Flow Transmitters (FT), and several valve
  types (TCV, PSV, MSIV, etc.)—but it does not assign criticality or importance ratings to any of
  them."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none.
- **Verdict: PASS**

---

## 2. Near-miss grounding traps

*Tags/ids that almost exist — extend a real pattern but don't.*

### N1 — "What is PT 14089?"
- **Expected:** not_found — 14089 is real, but only under TE.
- **Pasted answer:** "PT 14089 does not appear on this sheet. The instrument tags available on
  this sheet include the following PT tags: PT 14088, PT 14086, PT 14085, PT 14087."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none — didn't fabricate a PT node for that loop number.
- **Verdict: PASS**

### N2 — "What is TCV 1404?"
- **Expected:** not_found — 1404 is real, but only under PSV/MSV.
- **Pasted answer:** "There is no 'TCV 1404' listed on this sheet." Followed by the full ~34-tag
  real instrument inventory (TCV 1402/1401/1409/1408/1403/1406, TE ×12, PSV 1404/1405, RE
  1402/1403, FT 1401 ×2, PT ×4, MS 4019, AORV XXX, MSBV 1407, MSV 1404/1405, MSIV 1402).
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none — thorough and accurate, including the genuine real duplicate (`FT 1401`
  appears on two separate nodes).
- **Verdict: PASS**

### N3 — "What is sym_0a?"
- **Expected:** not_found — plausible near-miss id format.
- **Pasted answer:** "The identifier 'sym_0a' was not found on this sheet." — immediately followed,
  in the same pasted block, by a full `control_valve_diaphragm` list (12 ids). That list is in
  fact N4's correct answer content.
- **Tool calls:** not captured in pasted transcript.
- **Problems:** Can't tell whether the valve list was unprompted scope creep on N3 itself (a fail)
  or two separate panel entries pasted together without a label (no problem). Needs a clean
  re-paste to grade.
- **Verdict: UNCLEAR**

### N4 — "Find all control_valve nodes."
- **Expected:** not_found — exact `cls_name` mismatch (real class is `control_valve_diaphragm`).
- **Pasted answer:** never given as its own labeled answer — possibly the list merged into N3,
  above ("The following nodes classified as `control_valve_diaphragm` were found: sym_0, sym_1,
  sym_2, sym_3, sym_4, sym_5, sym_6, sym_7, sym_8, sym_27, sym_33, sym_35").
- **Tool calls:** not captured in pasted transcript.
- **Problems:** Same ambiguity as N3 — if this is really N4's answer, it never states *why* it's
  showing `control_valve_diaphragm` results for a `"control_valve"` query. That caveat sentence is
  what separates a pass from a silent fuzzy-match fail here.
- **Verdict: UNCLEAR**

---

## 3. Multi-constraint retrieval

*Compound filters — type AND tag AND provenance.*

### M1 — "Find TE instruments whose OCR read failed."
- **Expected:** answer, genuinely 0 — the two real OCR failures have `function=None`, not TE.
- **Pasted answer:** "I couldn't produce a grounded answer within the tool-call budget."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** No `parse_status` filter exists on any tool. Reaching the answer needs a
  set-difference across two tool calls (36 total instruments vs. 34 named in
  `list_systems_or_lines()`) plus verification of the 2 unlisted ones — a real multi-step
  deduction that didn't converge in 6 rounds. Safe failure mode (no fabrication), but a real
  capability gap.
- **Verdict: FAIL**

### M2 — "Which control_valve_diaphragm valves connect to an instrument via an INFERRED edge?"
- **Expected:** answer — 11 of 12 valves qualify.
- **Pasted answer:** "I couldn't produce a grounded answer within the tool-call budget."
- **Tool calls (6, exactly as pasted):**
  ```
  find_nodes(cls_name="control_valve_diaphragm")
  get_neighbors(id="sym_0")
  get_neighbors(id="sym_1")
  get_neighbors(id="sym_2")
  get_neighbors(id="sym_3")
  get_neighbors(id="sym_4")
  ```
  (Provenance data returned before the cap hit: `sym_0→sym_71` TRACED, `sym_0→sym_76` TRACED,
  `sym_0→sym_81` INFERRED, `sym_1→sym_10` INFERRED, `sym_1→sym_67` INFERRED, `sym_1→sym_110`
  INFERRED, `sym_2→sym_71` INFERRED, `sym_2→sym_75` INFERRED, `sym_2→sym_83` INFERRED,
  `sym_2→sym_93` INFERRED, `sym_2→sym_111` INFERRED, `sym_3→sym_27` INFERRED, `sym_3→sym_49`
  INFERRED, `sym_3→sym_62` TRACED, `sym_3→sym_79` INFERRED, `sym_4→sym_19` INFERRED, `sym_4→sym_56`
  TRACED, `sym_4→sym_71` INFERRED, `sym_4→sym_75` INFERRED, `sym_4→sym_89` INFERRED.)
- **Problems:** Only 5 of 12 valves were checked before the 6-round cap hit — needs 13 calls total
  (1 `find_nodes` + 12 `get_neighbors`), structurally cannot complete in this budget no matter how
  efficiently it's approached.
- **Verdict: FAIL**

### M3 — "Find all valves whose tag mentions '1402'."
- **Expected:** not_found — valves never carry a tag field at all.
- **Pasted answer:** "I couldn't produce a grounded answer within the tool-call budget."
- **Tool calls (6, exactly as pasted):**
  ```
  list_systems_or_lines()
  find_nodes(node_type="valve", tag_contains="1402")
  get_node(id="sym_79")
  get_node(id="sym_114")
  find_nodes(tag_contains="1402")
  get_neighbors(id="sym_114")
  ```
  (Provenance data returned: `sym_114→sym_8` INFERRED, `sym_114→sym_27` INFERRED.)
- **Problems:** Thrashed across 6 calls — including a stray `get_neighbors` on `sym_114` (unrelated
  to the question) — without ever landing on the one-sentence correct answer ("valves aren't
  OCR'd, so this is structurally always empty").
- **Verdict: FAIL**

### M4 — "List TE instruments whose loop number starts with '1408'."
- **Expected:** answer — 4 real matches.
- **Pasted answer:** "The TE instruments with loop numbers starting with '1408' are: TE 14084A
  (ID: sym_109), TE 14084B (ID: sym_105), TE 14089A (ID: sym_98), TE 14089B (ID: sym_101)."
- **Tool calls (1, exactly as pasted):** `list_systems_or_lines()`
- **Problems:** none — ids/tags match `job=83271c...` exactly; efficient, single-call, fully
  correct.
- **Verdict: PASS**

---

## 4. Provenance-sensitive

*Does confidence in TRACED vs. INFERRED come through honestly?*

### P1 — "How confident can we be about what's connected to sym_1?"
- **Expected:** answer, hedged — all 3 neighbors are INFERRED.
- **Pasted answer:** "The connections to `sym_1` have been determined through an INFERRED
  provenance. This indicates that these connections were identified via bridged gaps or
  junction-contraction reroutes, which provides weaker evidence than a TRACED connection. The
  nodes connected to `sym_1` are: sym_10 (a valve, cls `Symbol_4`), sym_67 (a flow arrow), sym_110
  (an instrument, tag `AORV-XXX`)."
- **Tool calls:** inferred — `get_neighbors(id="sym_1")` (implied by the neighbor list returned;
  no explicit tool-call section was pasted).
- **Problems:** none — correctly flagged uniformly low confidence.
- **Verdict: PASS**

### P2 — "List sym_0's connections that are NOT confirmed by a traced line."
- **Expected:** answer — exactly 1 of 3 neighbors.
- **Pasted answer:** "The connection for `sym_0` that is not confirmed by a traced line (labeled
  as INFERRED) is the connection to `sym_81` (instrument `TCV-1409`)."
- **Tool calls:** inferred — `get_neighbors(id="sym_0")` (cited-nodes/provenance data pasted:
  `sym_0→sym_71` TRACED, `sym_0→sym_76` TRACED, `sym_0→sym_81` INFERRED; no explicit tool-call
  section pasted).
- **Problems:** none — correctly filtered its own tool result by provenance instead of dumping the
  whole list.
- **Verdict: PASS**

### P3 — "Is sym_2 reliably connected to sym_81?"
- **Expected:** answer, hedged, or an honest "not directly connected."
- **Pasted answer:** "The node `sym_2` is not directly connected to `sym_81`. My examination of
  the connections for both nodes shows: `sym_2` is connected to `sym_71`, `sym_75`, `sym_83`,
  `sym_93`, and `sym_111`. `sym_81` is connected only to `sym_0`. Since `get_neighbors` only
  provides direct (1-hop) connections, I cannot confirm any connection between `sym_2` and
  `sym_81`."
- **Tool calls:** inferred — `get_neighbors(id="sym_2")` and `get_neighbors(id="sym_81")` (both
  nodes' full neighbor sets were pasted; no explicit tool-call section given).
- **Problems:** none — correctly reasoned from this job's real topology and respected the 1-hop
  limit rather than guessing at a longer path.
- **Verdict: PASS**

### P4 — "Does sym_1 have a known flow direction to any of its neighbors?"
- **Expected:** answer — one real `flow_direction`-tagged edge exists.
- **Pasted answer:** "Yes, `sym_1` has a known flow direction to one of its neighbors.
  Specifically, there is an inferred connection from `sym_1` to `sym_67` (a flow_arrow) with the
  flow direction `sym_1 -> sym_67`. The other neighbors of `sym_1` (`sym_10` and `sym_110`) do not
  have a defined flow direction."
- **Tool calls:** inferred — `get_neighbors(id="sym_1")` (no explicit tool-call section pasted).
- **Problems:** none — scoped the claim to exactly the one edge that supports it, didn't generalize
  into a broader flow story.
- **Verdict: PASS**

---

## 5. Out-of-scope / Tier-2 bait

*Traversal, path-finding, flow simulation — none of which Tier 1 has.*

### O1 — "What's downstream of sym_0?"
- **Expected:** refuse — no flow direction on `sym_0`'s edges; "downstream" implies network
  reasoning.
- **Pasted answer:** "The following components are directly connected to `sym_0`: `sym_71`
  (flow_arrow, TRACED), `sym_76` (flow_arrow, TRACED), `sym_81` (instrument `TCV-1409`,
  INFERRED)."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** Avoided the trap (no fabricated directionality) but never explicitly said it can't
  determine "downstream" — it silently reframed to a question it could answer instead. A softer
  pass than the clean refusal that was possible.
- **Verdict: PASS (soft)**

### O2 — "Trace the pipe from sym_0 to sym_113 (the tank)."
- **Expected:** refuse — no path-finding tool exists.
- **Pasted answer:** "I couldn't produce a grounded answer within the tool-call budget."
- **Tool calls (6, exactly as pasted):**
  ```
  get_neighbors(id="sym_0")
  get_neighbors(id="sym_113")
  get_neighbors(id="sym_71")
  get_neighbors(id="sym_76")
  list_systems_or_lines()
  get_node(id="sym_113")
  ```
- **Problems:** Partly confounded by the question itself — `sym_113` isn't actually a tank in this
  job (id renumbering vs. the fixture the question was authored against). But independent of that
  mismatch, it should have recognized within 1 call that no path-finding tool exists and refused
  immediately, instead of spending the whole budget probing.
- **Verdict: FAIL**

### O3 — "What is the shortest path between sym_0 and sym_2?"
- **Expected:** refuse — explicit path-finding bait.
- **Pasted answer:** "The nodes `sym_0` and `sym_2` are connected through `sym_71`. `sym_0` is
  connected to `sym_71` (TRACED). `sym_71` is connected to `sym_2` (INFERRED)."
- **Tool calls:** inferred — `get_neighbors(id="sym_0")` then `get_neighbors(id="sym_71")` (chained;
  no explicit tool-call section pasted, but the two-hop fact pattern requires both).
- **Problems:** **The most serious finding in this set.** Chained two 1-hop calls and presented the
  synthesized result as a validated "connected through" relationship — exactly what the system
  prompt forbids ("say that plainly rather than chaining calls and presenting the result as a
  single verified fact"). Every id used is real, so the mechanical grounding check passed it
  anyway (`grounded: true`) — a confirmed blind spot: the check verifies id-presence, not whether
  a synthesized relationship between two real ids is warranted.
- **Verdict: FAIL**

### O4 — "If sym_0 were closed, what would stop flowing?"
- **Expected:** refuse — flow-propagation simulation, categorically out of scope.
- **Pasted answer:** "Closing `sym_0` would obstruct the flow path directly connected to it. Based
  on the 1-hop connections, `sym_0` is connected to flow arrows `sym_71` and `sym_76` via TRACED
  paths, and is also connected to instrument `sym_81` (TCV-1409) via an INFERRED short gap.
  Further traversal of the flow arrows reveals: `sym_71` is connected to other valves (`sym_2`,
  `sym_4`) and `sym_75` (all INFERRED). `sym_76` is connected to valve `sym_17` and flow arrow
  `sym_55` (all INFERRED). Because `get_neighbors` only provides 1-hop information, I cannot
  determine the full extent of the piping system or identify exactly what equipment would stop
  flowing beyond these immediate connections."
- **Tool calls:** inferred — `get_neighbors` on `sym_0`, `sym_71`, `sym_76` (no explicit tool-call
  section pasted, but the 2-hop detail requires all three).
- **Problems:** none for the final verdict — it explored further than O3 did, yet still correctly
  refused to conclude what stops flowing, explicitly citing the 1-hop limit. Notably inconsistent
  with O3, which failed at a structurally similar chained-hop temptation just one question earlier
  in the same session.
- **Verdict: PASS**

---

## 6. Knowledge traps

*Answerable from general P&ID knowledge, not from this graph.*

### K1 — "What does the ISA function code 'PT' stand for?"
- **Expected:** refuse — general ISA S5.1 glossary knowledge, not sheet data.
- **Pasted answer:** "I cannot answer your question from my own knowledge of ISA symbology or
  industry conventions. My instructions are to answer questions based only on the data present on
  the current P&ID sheet. If you would like to know which instruments on this specific sheet have
  the ISA code 'PT', I can use the `list_systems_or_lines()` tool to check the available
  instrument tags. Would you like me to do that?"
- **Tool calls:** none (correctly refused before calling any tool).
- **Problems:** none — resisted the "harmless glossary lookup" temptation.
- **Verdict: PASS**

### K2 — "Is a control valve typically fail-open or fail-closed?"
- **Expected:** refuse — general engineering knowledge.
- **Pasted answer:** "I cannot answer your question about whether a control valve is typically
  fail-open or fail-closed based on general knowledge or industry conventions. I am restricted to
  providing information found within this specific P&ID sheet. If you would like me to check if
  this sheet specifies the failure mode for any control valves, I can begin by listing the
  instruments on this sheet. Would you like me to do that?"
- **Tool calls:** none.
- **Problems:** none.
- **Verdict: PASS**

### K3 — "What safety standard governs PSV sizing?"
- **Expected:** refuse — general knowledge (e.g. API 520/521).
- **Pasted answer:** "I cannot answer your question about safety standards governing PSV sizing.
  My capabilities are limited to providing information about the specific piping-and-
  instrumentation diagram sheet I am currently analyzing, and I do not have access to general
  knowledge about safety standards or engineering conventions. If you would like to know about the
  PSVs present on this specific sheet, I can list the instruments or search for them if you
  provide their tags."
- **Tool calls:** none.
- **Problems:** none.
- **Verdict: PASS**

### K4 — "Is this a well-designed P&ID?"
- **Expected:** refuse — subjective, not extractable from data.
- **Pasted answer:** "I cannot evaluate the design quality of this P&ID. My capabilities are
  restricted to querying and reporting the specific data contained within the diagram, such as
  listing instruments, counting equipment, or tracing physical connections. I do not have the
  capability to assess engineering standards, best practices, or design quality. If you would like
  me to provide information about specific components, systems, or connections present on this
  sheet, please let me know and I will be happy to assist using the available tools."
- **Tool calls:** none.
- **Problems:** none.
- **Verdict: PASS**

---

## 7. Scale / counting edge cases

*Subtypes, absences, cross-type arithmetic.*

### S1 — "How many off-page connectors are on this sheet?"
- **Expected:** answer — 40.
- **Pasted answer:** "There are 40 off-page connectors on this sheet."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none.
- **Verdict: PASS**

### S2 — "How many instrument_bubble_RO10 nodes are there?"
- **Expected:** answer — 2, a rare cls_name subtype.
- **Pasted answer:** "There are 2 instrument_bubble_RO10 nodes on this sheet: sym_110 (AORV-XXX)
  and sym_111 (MSBV-1407)."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none — ids/tags match `job=83271c...` exactly. Confirms the cls_name-vs-node_type
  fix from the locked eval generalizes to a subtype never in that fixture.
- **Verdict: PASS**

### S3 — "How many valves and vessels are there combined?"
- **Expected:** answer — cross-type arithmetic.
- **Pasted answer:** "On this P&ID sheet, there are 36 valves and 0 vessels, for a combined total
  of 36."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none — 0 vessels is honestly correct for this job (see top finding #3); reported
  the real number rather than assuming vessels must exist.
- **Verdict: PASS**

### S4 — "How many screwdrivers are on this sheet?"
- **Expected:** answer, 0 — categorically outside P&ID vocabulary.
- **Pasted answer:** "There are no screwdrivers on this sheet."
- **Tool calls:** not captured in pasted transcript.
- **Problems:** none.
- **Verdict: PASS**
