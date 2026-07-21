"""Phase 3 Task 3 — evaluate instrument-bubble tag OCR against docs/phase3_eval/tags_gt.csv.

For sheets 0, 3, 10:
  1. Build the DEDUPED node set (roi_filter -> centroid_nms(group_key=scored_family_group_key)
     -> build_node_set -> assert_no_duplicate_scored_nodes), exactly as production does.
  2. Run pidetect.text.ocr.run_ocr_on_nodes on the raw (pre-erasure) image.
  3. Match predicted instrument nodes to GT instrumentation nodes (same CtrMt@50%
     convention as scripts/measure_bubble_bbox_ratio.py).
  4. Score against docs/phase3_eval/tags_gt.csv:
       - matched + label_status in {ok, placeholder} -> scored normally
       - matched + label_status == unreadable -> excluded from the denominator,
         reported as its own count (a bubble no human can read is not an OCR failure)
       - unmatched GT node -> counts as a failure (not excluded)
     Metrics: exact-match accuracy (normalized reconstructed tag vs GT raw_tag) and
     micro-averaged CER (tag_raw_text vs GT raw_tag), both split by tag_parse_status.
  5. Separately: locate the 2 known non-instrument false-positive detections (hexagonal
     "DET A/SHT 2", "DET B/SHT 2", sheet 3) and report whether parse validation
     rejected them (tag_parse_status == "failed").

Writes docs/phase3_results.md.

Usage:
    PYTHONPATH=src python scripts/run_phase3_ocr_eval.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import (
    roi_filter, centroid_nms, build_node_set,
    scored_family_group_key, assert_no_duplicate_scored_nodes,
)
from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
PHASE4_CFG_PATH = REPO / "configs" / "phase4.yaml"
PHASE3_CFG_PATH = REPO / "configs" / "phase3.yaml"
GT_CSV_PATH = REPO / "docs" / "phase3_eval" / "tags_gt.csv"
SHEETS = [0, 3, 10]

# The 2 confirmed non-instrument false positives (docs/phase3_design.md §4), by bbox,
# so we can locate and OCR them too even though they have no GT counterpart.
FALSE_POSITIVE_BBOXES = {
    3: [(723, 1249, 767, 1289), (724, 914, 769, 954)],
}


def _load_gt_rows() -> dict[tuple[str, str], dict]:
    with open(GT_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["sheet_id"], r["gt_node_id"]): r for r in rows}


def _load_gt_instrumentation(gml: Path) -> list[tuple[str, float, float, float, float]]:
    import networkx as nx
    g = nx.read_graphml(str(gml))
    return [
        (nid, float(a["xmin"]), float(a["ymin"]), float(a["xmax"]), float(a["ymax"]))
        for nid, a in g.nodes(data=True) if a.get("label") == "instrumentation"
    ]


def _match(pred_nodes, gt_nodes):
    """Greedy CtrMt@50%, sorted by confidence desc. Returns list of (pred_node, gt_tuple)."""
    preds_sorted = sorted(pred_nodes, key=lambda n: -n.conf)
    used_gt = set()
    matches = []
    for p in preds_sorted:
        best_d, best_gt = float("inf"), None
        for gt in gt_nodes:
            gid, gx1, gy1, gx2, gy2 = gt
            if gid in used_gt:
                continue
            gw, gh = gx2 - gx1, gy2 - gy1
            thr = 0.5 * math.sqrt(max(gw * gh, 1.0))
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            d = math.hypot(p.cx - gcx, p.cy - gcy)
            if d <= thr and d < best_d:
                best_d, best_gt = d, gt
        if best_gt is not None:
            used_gt.add(best_gt[0])
            matches.append((p, best_gt))
    return matches


def _normalize(function: str | None, loop_number: str | None) -> str:
    if function is None or loop_number is None:
        return ""
    return f"{function} {loop_number}".strip().upper()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(PHASE4_CFG_PATH) as f:
        p4cfg = yaml.safe_load(f)
    with open(PHASE3_CFG_PATH) as f:
        p3cfg = yaml.safe_load(f)
    p3cfg["nms_centroid_frac"] = p4cfg["nms_centroid_frac"]

    weights = REPO / args.weights
    cache_dir = REPO / args.cache_dir

    try:
        from pidetect.text.ocr import run_ocr_on_nodes
    except ImportError as exc:
        print(f"FATAL: cannot import pidetect.text.ocr -- {exc}")
        print("PaddleOCR/paddlepaddle is not installed in this environment. "
              "See requirements.txt for the Python-version compatibility note.")
        sys.exit(1)

    gt_rows = _load_gt_rows()

    all_scored = []  # dicts with sheet_id, gt_node_id, gt_row, pred_node (or None)
    fp_results = []  # (sheet_id, bbox, tag_parse_status, tag_raw_text)

    for sheet_id in SHEETS:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        bg_bboxes, img_w, img_h, _ = _load_bg_bboxes(gml)
        raw_preds = _load_or_infer(sheet_id, png, weights, args.conf, args.iou, args.device, cache_dir)
        preds_after_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=p4cfg["roi_border_margin_frac"],
        )
        preds_deduped, _ = centroid_nms(
            preds_after_roi, centroid_frac=p4cfg["nms_centroid_frac"], group_key=scored_family_group_key,
        )
        nodes = build_node_set(preds_deduped)
        assert_no_duplicate_scored_nodes(nodes, centroid_frac=p4cfg["nms_centroid_frac"])

        img_bgr = cv2.imread(str(png))
        try:
            run_ocr_on_nodes(img_bgr, nodes, p3cfg)
        except ImportError as exc:
            print(f"\nFATAL: PaddleOCR engine unavailable -- {exc}")
            print("paddleocr/paddlepaddle could not be imported (see requirements.txt "
                  "for the Python-version compatibility note). No OCR was run; no "
                  "results to score. Cannot proceed with Task 3 in this environment.")
            sys.exit(1)

        instrument_nodes = [n for n in nodes if n.node_type == "instrument"]
        gt_instr = _load_gt_instrumentation(gml)
        matches = _match(instrument_nodes, gt_instr)
        matched_gt_ids = {gt[0] for _, gt in matches}
        matched_by_gt = {gt[0]: pred for pred, gt in matches}

        for gid, _, _, _, _ in gt_instr:
            row = gt_rows.get((str(sheet_id), gid))
            if row is None:
                print(f"  WARNING: sheet {sheet_id} GT node {gid} has no tags_gt.csv row -- skipping")
                continue
            pred = matched_by_gt.get(gid)
            all_scored.append({"sheet_id": sheet_id, "gt_node_id": gid, "gt_row": row, "pred": pred})

        # False positives: locate by bbox proximity among instrument_nodes, OCR already ran.
        for x1, y1, x2, y2 in FALSE_POSITIVE_BBOXES.get(sheet_id, []):
            ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
            best, bestd = None, 1e9
            for n in instrument_nodes:
                d = math.hypot(n.cx - ccx, n.cy - ccy)
                if d < bestd:
                    bestd, best = d, n
            if best is not None and bestd < 30:
                fp_results.append((sheet_id, (x1, y1, x2, y2), best.tag_parse_status, best.tag_raw_text))
            else:
                fp_results.append((sheet_id, (x1, y1, x2, y2), "NOT_FOUND_IN_CURRENT_PREDICTIONS", ""))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    status_counts = Counter()
    exact_match_by_status = defaultdict(lambda: [0, 0])  # status -> [n_correct, n_total]
    cer_edit_sum = 0
    cer_char_sum = 0
    n_unreadable = 0
    n_unmatched_gt_failure = 0

    detail_rows = []

    for item in all_scored:
        row = item["gt_row"]
        label_status = row["label_status"]
        gt_raw = row["raw_tag"]

        if label_status == "unreadable":
            n_unreadable += 1
            continue

        pred = item["pred"]
        if pred is None:
            n_unmatched_gt_failure += 1
            status_counts["UNMATCHED_GT"] += 1
            detail_rows.append((item["sheet_id"], item["gt_node_id"], gt_raw, "(no matching prediction)", "UNMATCHED_GT", False))
            continue

        status = pred.tag_parse_status
        status_counts[status] += 1

        pred_normalized = _normalize(pred.tag_function, pred.tag_loop_number)
        gt_normalized = gt_raw.strip().upper()
        correct = (pred_normalized == gt_normalized) and pred_normalized != ""

        exact_match_by_status[status][1] += 1
        if correct:
            exact_match_by_status[status][0] += 1

        edits = _levenshtein(pred.tag_raw_text.strip().upper(), gt_raw.strip().upper())
        cer_edit_sum += edits
        cer_char_sum += max(1, len(gt_raw.strip()))

        detail_rows.append((item["sheet_id"], item["gt_node_id"], gt_raw, pred_normalized, status, correct))

    n_scoreable = sum(v[1] for v in exact_match_by_status.values()) + n_unmatched_gt_failure
    n_correct_total = sum(v[0] for v in exact_match_by_status.values())
    overall_accuracy = n_correct_total / n_scoreable if n_scoreable else 0.0
    ok_statuses = {"ok", "ok_placeholder"}
    n_ok_total = sum(v[1] for s, v in exact_match_by_status.items() if s in ok_statuses)
    n_ok_correct = sum(v[0] for s, v in exact_match_by_status.items() if s in ok_statuses)
    ok_accuracy = n_ok_correct / n_ok_total if n_ok_total else 0.0
    micro_cer = cer_edit_sum / cer_char_sum if cer_char_sum else 0.0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    lines.append("# Phase 3 Results — Instrument-Bubble Tag OCR Evaluation\n")
    lines.append(f"**Sheets:** {SHEETS}  **GT eval set:** `docs/phase3_eval/tags_gt.csv` (106 rows)\n")
    lines.append("## Summary\n")
    lines.append(f"- Scoreable rows (excludes {n_unreadable} unreadable): **{n_scoreable}**")
    lines.append(f"- Overall exact-match accuracy: **{overall_accuracy:.1%}** ({n_correct_total}/{n_scoreable})")
    lines.append(f"- Exact-match accuracy on ok/ok_placeholder rows: **{ok_accuracy:.1%}** ({n_ok_correct}/{n_ok_total})")
    lines.append(f"- Micro-averaged CER: **{micro_cer:.4f}** ({cer_edit_sum} edits / {cer_char_sum} GT chars)")
    lines.append(f"- Unmatched GT (scored as failure): {n_unmatched_gt_failure}")
    lines.append(f"- Unreadable GT rows (excluded from denominator): {n_unreadable}\n")

    lines.append("## Gate check\n")
    gate_ok_pass = "PASS" if ok_accuracy >= 0.90 else "FAIL"
    gate_overall_pass = "PASS" if overall_accuracy >= 0.80 else "FAIL"
    lines.append(f"- >=90% exact-match on ok rows: {ok_accuracy:.1%} -> **{gate_ok_pass}**")
    lines.append(f"- >=80% exact-match overall: {overall_accuracy:.1%} -> **{gate_overall_pass}**\n")

    lines.append("## Split by tag_parse_status\n")
    lines.append("| Status | n | exact-match |")
    lines.append("|---|---|---|")
    for status, (correct, total) in sorted(exact_match_by_status.items()):
        acc = correct / total if total else 0.0
        lines.append(f"| {status} | {total} | {acc:.1%} ({correct}/{total}) |")
    if n_unmatched_gt_failure:
        lines.append(f"| UNMATCHED_GT | {n_unmatched_gt_failure} | 0.0% (0/{n_unmatched_gt_failure}) |")

    lines.append("\n## Known non-instrument false positives (DET A / DET B, sheet 3)\n")
    n_rejected = sum(1 for _, _, status, _ in fp_results if status == "failed")
    lines.append(f"Correctly rejected by parse validation: **{n_rejected}/{len(fp_results)}**\n")
    for sheet_id, bbox, status, raw in fp_results:
        lines.append(f"- sheet {sheet_id} bbox={bbox}: tag_parse_status=`{status}` raw_text={raw!r}")

    out_path = REPO / "docs" / "phase3_results.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
