# Phase 3 Results — Instrument-Bubble Tag OCR Evaluation

**Sheets:** [0, 3, 10]  **GT eval set:** `docs/phase3_eval/tags_gt.csv` (106 rows)

## Summary

- Scoreable rows (excludes 1 unreadable): **105**
- Overall exact-match accuracy: **83.8%** (88/105)
- Exact-match accuracy on ok/ok_placeholder rows: **90.7%** (88/97)
- Micro-averaged CER: **0.0241** (22 edits / 913 GT chars)
- Unmatched GT (scored as failure): 0
- Unreadable GT rows (excluded from denominator): 1

## Gate check

- >=90% exact-match on ok rows: 90.7% -> **PASS**
- >=80% exact-match overall: 83.8% -> **PASS**

## Split by tag_parse_status

| Status | n | exact-match |
|---|---|---|
| failed | 8 | 0.0% (0/8) |
| ok | 94 | 90.4% (85/94) |
| ok_placeholder | 3 | 100.0% (3/3) |

## Known non-instrument false positives (DET A / DET B, sheet 3)

Correctly rejected by parse validation: **2/2**

- sheet 3 bbox=(723, 1249, 767, 1289): tag_parse_status=`failed` raw_text='B DET 2 SHT'
- sheet 3 bbox=(724, 914, 769, 954): tag_parse_status=`failed` raw_text='A DET 2 SHT'

## Ground-truth correction

Sheet 3 `instrumentation32`, `34`, `35`, `78` were hand-labeled `function=UT`; a visual
re-check of the crops corrected this to `function=LIT`. Evidence: glyph column-profile on
the crops shows three ink cells of widths ~21px / 7px / 21px — the narrow 7px middle cell
is an "I", giving L‑I‑T (a "UT" would produce only two cells). At small scale the L's foot
fuses with the I's stem and reads as a single "U" stroke to the eye — a genuine optical
ambiguity in the source diagram, not a detector or labeling carelessness error. All other
disputed rows from the pre-correction mismatch list (leading-digit drops, B→8, B→6, A→9
substitutions) were re-checked and the original labels were confirmed correct — those are
genuine OCR misreads, not label errors.

Notably, OCR itself surfaced this disagreement: the model consistently read these 4 bubbles
as "LIT ..." against a GT that said "UT ...", and re-inspection showed the model had read
the ambiguous glyph correctly where the original human labeler had not.

**Before vs after correction (same 3 sheets, same OCR run, GT only changed):**

| Metric | Pre-correction | Post-correction |
|---|---|---|
| Exact-match on ok/ok_placeholder rows | 89.7% (87/97) | **90.7% (88/97)** |
| Exact-match overall | 82.9% (87/105) | 83.8% (88/105) |
| Micro-averaged CER | 0.0330 (30/909) | 0.0241 (22/913) |
| Gate (≥90% ok rows) | FAIL | **PASS** |
| Gate (≥80% overall) | PASS | PASS |

Only 1 of the 4 corrected rows flips from incorrect to correct (`instrumentation78`: OCR
read `'LIT 165108A'`, an exact match to the corrected GT). The other 3
(`instrumentation32/34/35`) still have independent digit-read errors in the loop number
and remain mismatches even under the corrected function — see the full list below.

## All remaining mismatches (17)

| Sheet/Node | gt_raw_tag → ocr_raw_text | Edit distance |
|---|---|---|
| 0/instrumentation18 | `PT 14087` → `PT 4087` | 1 |
| 0/instrumentation33 | `MS 14010` → `MS 4019` | 2 |
| 0/instrumentation61 | `TE 14093A` → `TE 4093A` | 1 |
| 3/instrumentation28 | `TE 170117B` → `TE 1701178` | 1 |
| 3/instrumentation32 | `LIT 165108B` → `LIT 1651088` | 1 |
| 3/instrumentation34 | `LIT 165109A` → `LIT 65109A` | 1 |
| 3/instrumentation35 | `LIT 165109B` → `LIT 1651096` | 1 |
| 3/instrumentation68 | `TE 170135A` → `TE 70135A` | 1 |
| 3/instrumentation71 | `TE 170135B` → `TE 1701358` | 1 |
| 3/instrumentation83 | `TE 165110A` → `TE 65119A` | 2 |
| 3/instrumentation85 | `TE 165110B` → `TE 1651108` | 1 |
| 3/instrumentation88 | `TE 160105B` → `TE 1601058` | 1 |
| 10/instrumentation77 | `MOV 1115` → `MOV 1115 4` | 2 |
| 10/instrumentation82 | `PIT 11256` → `PIT 1 1256` | 1 |
| 10/instrumentation84 | `LIT 11857` → `LIT 1 1857` | 1 |
| 10/instrumentation88 | `PIT 11561` → `PIT 1156y` | 1 |
| 10/instrumentation89 | `LIT 11858` → `LIT 1858 1` | 3 |
