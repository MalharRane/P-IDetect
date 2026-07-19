# Phase 4 Step 4 — Scoped Design

**Date:** 2026-07-17  
**Basis:** step3_final.md FP breakdown; step4_design.md §3–4 (spatial veto, now disproven);
crossing census (crossings 248–553 px off the straight A→B segment)  
**Goal:** suppress the 39 S4 FPs; do NOT touch P3 or OTHER FPs; hold recall ≥ 0.75

---

## 1. Why the Previous Design (Spatial Veto) Doesn't Work

`phase4_step4_design.md §4a` proposed: veto short-gap pair (A, B) if a detected
crossing centroid projects onto the A→B segment within t ∈ [0.10, 0.90] and perp ≤ 15 px.

The census disproves this. For the actual S4 FPs, the separating crossing is **248–553 px
off the straight A→B segment**. The crossing is NOT between the two symbols on the direct
line; it sits on a different part of the pipe network that happens to route A toward B via
a long detour. No reasonable perp tolerance catches it, and a wide tolerance would
catch coincidental crossings on unrelated pipes.

The spatial veto is the wrong primitive. The correct question is topological: "does the pipe
network connect A to B WITHOUT going through a crossing?"

---

## 2. Why Short-Gap Edges Bypass Contraction (Architecture Summary)

`run_step5` / `build_graph` inserts short-gap pairs as **direct sym_A ↔ sym_B edges**
with no intermediate node (see [build.py:176–186](src/pidetect/graph/build.py#L176-L186)):

```python
for id_a, id_b in s3.short_gap_pairs:
    G.add_edge(gid_a, gid_b, branch_type=-1, short_gap=True)
```

`run_step6` contracts connector and crossing **nodes**. A direct edge has no intermediate
node to contract. Contraction never sees this edge, so it survives even when A and B are
crossing-separated in the pipe topology.

---

## 3. The New Mechanism: Contracted-Skeleton Crossing-Separation Test

### 3a. Algorithm

Build two auxiliary graphs that contain ONLY skeleton-traced edges (no short-gap):

**SR** — raw skeleton graph (junction nodes present)  
**Sc** — SR after contracting connectors and crossings with pass-through semantics

For each short-gap candidate pair (A, B):

| `has_path(SR, A, B)` | `has_path(Sc, A, B)` | Meaning | Decision |
|---|---|---|---|
| False | — | No skeleton path at all; pipe was erased | **KEEP** (erased short-pipe TP) |
| True | True | Skeleton path exists and survives contraction | **KEEP** (genuine connection) |
| True | False | Skeleton path exists but is broken by crossing contraction | **SUPPRESS** (S4 FP) |

The crossing-separation test fires exactly in the third row: A and B are reachable in the
raw skeleton, but that path passes through a crossing whose pass-through semantics disconnect
them. After suppression the short-gap edge is simply not inserted.

### 3b. Data Flow

```
Step 3  run_step3(...)
        → s3: branches, endpoints, short_gap_pairs

Step 4  run_step4(s3.skeleton, s3.branches, s3.endpoints, ...)
        → s4: connectors (with branch_ids), crossings (with pass_through_pairs, branch_ids)

[NEW]   Build SR:
        Call build_graph(all_nodes, off_page, s3_no_sg, s4)
          where s3_no_sg = replace(s3, short_gap_pairs=[])
          i.e. skeleton branches and junction nodes are added; short-gap edges are NOT.
        SR is a nx.Graph with sym_N and junc_J nodes; branch edges only.

[NEW]   Build Sc:
        Sc = SR.copy()
        _contract_connectors(Sc)   # existing function — all-pairs edges, remove connector
        _contract_crossings(Sc)    # existing function — pass-through pairs only, remove crossing
        Sc now has only symbol nodes; crossing-separated components are disconnected.

[NEW]   Veto pass (before run_step5):
        suppressed = set()
        for (id_a, id_b) in s3.short_gap_pairs:
            gid_a, gid_b = sym_gid(id_a), sym_gid(id_b)
            sr_conn = nx.has_path(SR, gid_a, gid_b) if gid_a in SR and gid_b in SR else False
            sc_conn = nx.has_path(Sc, gid_a, gid_b) if gid_a in Sc and gid_b in Sc else False
            if sr_conn and not sc_conn:
                suppressed.add((id_a, id_b))

        s3_vetoed = replace(s3, short_gap_pairs=[
            p for p in s3.short_gap_pairs if p not in suppressed
        ])

Step 5  run_step5(all_nodes, off_page, s3_vetoed, s4)
        → crossing-vetoed pairs are absent; they were never inserted

Step 6  run_step6(s5)  (unchanged)
        → contracted symbol-to-symbol graph
```

Functions needed: none new beyond the two existing `_contract_*` functions already in
[build.py:235–303](src/pidetect/graph/build.py#L235-L303). The veto is ~15 lines of
pure graph logic callable from `scripts/run_phase4_steps03.py`.

New config param: none (the test is binary; no tolerance needed because connectivity in
a graph is exact).

---

## 4. FP Coverage: What This Reaches and What It Doesn't

The S4 FPs split into two structurally different populations:

**Population A — short-gap S4 FPs (SG?=yes): 31 of 39 total**

| Sheet | Count | This mechanism reaches |
|---|---|---|
| 0 | 25 | ✓ Targeted — all 25 are short-gap |
| 3 | 3 | ✓ Targeted — (sym_12,sym_54), (sym_5,sym_54), (sym_54,sym_74) |
| 10 | 3 | ✓ Targeted — all 3 sheet-10 S4 FPs are short-gap |

The mechanism fires iff the separating crossing is (a) detected by step4 and (b) its
`pass_through_pairs` correctly excludes A and B. Upper bound = suppress all 31. Actual
suppression depends on step4 crossing detection recall, addressed in §5.

**Population B — skeleton-traced S4 FPs (SG?=no): 8 of 39 total**

| Sheet | Count | Pairs | This mechanism reaches |
|---|---|---|---|
| 0 | 2 | (sym_21,sym_35), (sym_30,sym_36) | ✗ Not targeted |
| 3 | 6 | (sym_0,sym_1), (sym_0,sym_11), (sym_0,sym_74), (sym_1,sym_11), (sym_1,sym_74), (sym_11,sym_74) | ✗ Not targeted |

For skeleton-traced pairs, a direct skeleton branch connects A and B (that is how the edge
exists — not via short-gap). In SR, A and B are **directly connected** by that branch edge.
Sc preserves direct edges (contraction only removes junction nodes, not direct symbol-to-symbol
branches). Therefore `sc_conn = True` for all Population-B pairs → the veto does not fire.

These 8 FPs are a separate problem (likely branch-endpoint binding bleed onto incorrect
symbols). They are OUT OF SCOPE for this mechanism and are noted as an open residual in §8.

---

## 5. Predicted Suppression per Sheet

**Sheet 0 (25 short-gap S4 FPs, Population A)**

Sheet 0 has 71 GT crossing nodes. The S4 FPs are:
- Symbol_4↔CVD clusters: (sym_10,sym_31), (sym_12,sym_30), (sym_12,sym_80),
  (sym_14,sym_33), (sym_14,sym_88), (sym_8,sym_27), (sym_8,sym_82)
- Valve_handwheel clusters: (sym_20,sym_21), (sym_20,sym_35), (sym_20,sym_107),
  (sym_21,sym_23), (sym_21,sym_87), (sym_21,sym_104), (sym_22,sym_35), (sym_22,sym_101),
  (sym_23,sym_28), (sym_23,sym_107), (sym_23,sym_115)
- CVD↔IB direct adjacency: (sym_35,sym_98), (sym_35,sym_113), (sym_36,sym_105),
  (sym_87,sym_115), (sym_92,sym_115), (sym_101,sym_103), (sym_105,sym_113)

These are spatially adjacent pairs on DIFFERENT process pipes, connected in the network
via crossings 248–553 px from the A→B line. Their pipe stubs should connect through the
main pipe network to the separating crossing. Predicted suppression:
- **Optimistic (step4 detects most crossings):** 20–25 suppressions → FP sheet 0: 34→9–14
- **Conservative (step4 recall ~60%):** 12–15 suppressions → FP sheet 0: 34→19–22

The valve_handwheel cluster (multiple handwheel symbols on the same pipe section, likely
operating different valves on a shared process line) will yield most suppressions if the
shared crossing is detected; failure to detect ONE crossing can block several suppressions.

**Sheet 3 (3 short-gap S4 FPs, Population A + 6 skeleton-traced, Population B)**

Short-gap targets: (sym_12,sym_54), (sym_5,sym_54), (sym_54,sym_74). All involve sym_54
(instrument_bubble). Their GT path goes through a crossing. If the separating crossing is
detected: up to 3 suppressions. Expected: 2–3.

Population B (sym_0, sym_1, sym_11, sym_74 cluster): 0 suppressions. These 6 remain.

Net sheet-3 FP: **13 → 10–11** (from 3 Population-A suppressions; 6 Population-B remain).

**Sheet 10 (3 short-gap S4 FPs — control sheet)**

Sheet 10's 3 S4 FPs are (sym_155,sym_185), (sym_159,sym_177), (sym_159,sym_190) — all
instrument_bubble↔instrument_bubble. If their separating crossings are detected: up to 3
suppressions. Sheet 10's dominant FP bucket is P3 (10/17 FPs) — the contracted-skeleton
veto never touches P3 pairs (those are pairs where NO GT path exists; sr_conn is likely
False → veto does not fire). The P3 count stays fixed.

Expected net FP change on sheet 10: **17 → 14–17** (0–3 from S4; 10 P3 unchanged).

---

## 6. TP Safety Analysis

### 6a. The General Safety Argument

The veto fires only when `sr_conn=True AND sc_conn=False`. A short-gap TP that is
**erased** (original bboxes touching, no skeleton in the corridor) has `sr_conn=False`
(nothing connects them through the skeleton) → the veto can never fire → SAFE.

All 8 of the 9 sheet-3 TPs are fully erased (orig_gap=0, 0 corridor skeleton pixels,
confirmed by trace_check2.py). The 9th (sym_14↔sym_8) has ~17 px of skeleton in the
gap — but this stub is so short it forms an isolated fragment unlikely to reach back
through the full pipe network to close a loop. These are trivially safe.

### 6b. The Actual Risk

The risk is subtler: the erased TPs still have **outbound skeleton stubs** that connect
into the main pipe network. If A's stub and B's stub both reach the same connected
skeleton component, `sr_conn=True`. If that component contains a detected crossing on the
path between A and B, `sc_conn=False` → wrongly suppressed.

Concretely: if A (CVD) has a left stub going to the process pipe, and B (IB) has a stub
going to the same process pipe, they ARE connected in SR via the process pipe network.
Whether `sc_conn=False` depends on whether a detected crossing lies on the skeleton path
between A's stub endpoint and B's stub endpoint in SR.

For genuine TP pairs, the GT contracted graph has a direct A-B edge. This means the GT
pipe topology connects A to B WITHOUT going through a separating crossing. The skeleton
should reflect the same topology. However, there are two failure modes:
1. Step4 produces a **false crossing** on the A→B path in the skeleton → sc_conn=False → wrong suppression.
2. Step4 detects a **real crossing** elsewhere on the path that happens to disconnect A from B in Sc, even though the GT considers them directly connected (possible if the GT contracted graph routes through the crossing's pass-through, but the skeleton routes differently).

### 6c. Validation Plan Before Coding

For each of the 9 sheet-3 TPs, compute `(sr_conn, sc_conn)` on the ACTUAL SR and Sc
graphs from step4 output, using the current crossing detection parameters. This is a
read-only measurement — no production change, no new mechanism:

```python
s3 = run_step3(...)
s4 = run_step4(...)
SR = build_graph(all_nodes, off_page, replace(s3, short_gap_pairs=[]), s4)
Sc_copy = SR.copy(); _contract_connectors(Sc_copy); _contract_crossings(Sc_copy)

for (gtu, gtv) in sheet3_tp_gt_pairs:
    pred_u, pred_v = g2p[gtu], g2p[gtv]   # GT→pred node mapping
    gid_a, gid_b = sym_gid(pred_u), sym_gid(pred_v)
    sr = nx.has_path(SR, gid_a, gid_b)
    sc = nx.has_path(Sc_copy, gid_a, gid_b)
    print(f"{pred_u} ↔ {pred_v}: sr={sr}, sc={sc}",
          "SUPPRESS_RISK" if sr and not sc else "SAFE")
```

**Expected safe result**: `sr=False` for the 8 fully-erased TPs (no skeleton path at all)
→ veto cannot fire. OR `sc=True` for any TP where the skeleton path does not route through
a detected crossing.

**Flag (stop before implementation)**: if ANY TP shows `sr=True AND sc=False`, that TP is
a recall risk. Document which crossing on the skeleton path is responsible and decide whether
to: (a) exclude that crossing node from Sc construction (mark it as unconfident), or (b)
add a bypass rule (if the pair's gt_edge exists in GT contracted graph, never suppress).

The gate for proceeding to code: all 9 sheet-3 TPs must show `sr=False OR sc=True`.

---

## 7. Sheet 10 as Control

Sheet 10 has P=0.452, R=0.636, F1=0.528. Its FP breakdown is:
- S4: 3 (17%) — targeted by this mechanism
- P3: 10 (58%) — NOT targeted (no GT path; sr_conn=False for these)
- OTHER: 4 (23%) — NOT targeted

The contracted-skeleton veto will NOT touch the P3 FPs because for a P3 pair there is no GT
path; in the skeleton, A and B are likely NOT connected at all (sr_conn=False → veto bypassed).
Sheet 10 therefore acts as a natural control: if the veto mechanism is correct, sheet-10 FP
count should change by at most 3 (the S4 sub-bucket) and recall should be unchanged.

Any recall change on sheet 10 after the veto is a red flag — it signals the veto is suppressing
TPs, not FPs.

---

## 8. Open Residuals Not Addressed by This Scope

| Residual | Count | Root cause | Needed fix |
|---|---|---|---|
| Population-B skeleton-traced S4 FPs | 8 total (6 sheet-3, 2 sheet-0) | Direct skeleton branch endpoint binds to a wrong symbol (branch runs past A and B but belongs to a different pipe) | Branch attribution: check that branch midpoint lies geometrically between A and B, not off to the side. Separate design pass. |
| P3 signal-line FPs | 19 total | Dashed/signal lines with no GT process-pipe connection; short-gap fires on the signal ink | Phase-3 line-type classification |
| Sheet-3 FNs (LONG bucket) | 1 (sym_10↔something) | dil_gap > 50 px; no short-gap candidate | Longer skeleton tracing or junction-merge |

The Population-B residual is structurally distinct and needs its own scoped design (branch
midpoint check). It accounts for all 6 skeleton-traced S4 FPs on sheet 3, which is why even
an ideal contracted-skeleton implementation leaves sheet-3 at ~10 FPs (the 6 skeleton-traced
plus 1 OTHER plus 3 P3).

---

## 9. Summary

| Question | Answer |
|---|---|
| Why does spatial veto fail? | Crossings are 248–553 px off the A→B line; they're on the pipe network route, not the direct segment |
| What is the correct discriminator? | Graph connectivity: `sr_conn=True AND sc_conn=False` |
| What does SR→Sc require? | `build_graph` without `short_gap_pairs` + `_contract_connectors` + `_contract_crossings` — all existing functions |
| How many S4 FPs targeted? | 31 of 39 (Population A, short-gap) |
| How many NOT targeted? | 8 (Population B, skeleton-traced) — separate problem |
| Predicted FP reduction (optimistic)? | 34→9–14 (sheet 0), 13→10–11 (sheet 3), 17→14–17 (sheet 10) |
| Recall risk? | Erased TPs (8/9 on sheet 3) are safe (sr_conn=False). One potential risk: TP where skeleton path routes through a false/misassigned detected crossing. Must validate all 9 TPs against SR/Sc before coding. |
| Gate to proceed? | All 9 sheet-3 TPs: `sr=False OR sc=True`. Any violation → stop and diagnose crossing detection FP. |
