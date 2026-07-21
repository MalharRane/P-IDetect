"""Phase 3 Task 1 — render all GT instrument-bubble crops for hand-labeling, BEFORE
any OCR runs (avoids anchoring the human labeler on model output).

For sheets 0, 3, 10:
  1. Crop every GT `instrumentation` node's own bbox + crop_padding_px (configs/phase3.yaml),
     white-fill any OTHER GT node's bbox (any label except "background") that falls inside
     the padded region, upscale by upscale_factor (LANCZOS), save to
     docs/phase3_eval/crops/sheet{N}_{gt_node_id}.png.
  2. Write docs/phase3_eval/tags_gt.csv: one row per GT instrumentation node, sorted in
     reading order (top-to-bottom, left-to-right) per sheet, with function/loop_number/
     raw_tag columns EMPTY for hand-labeling.
  3. Re-run the sheet-level predicted-bubble matching (same convention as
     scripts/measure_bubble_bbox_ratio.py) and crop every UNMATCHED predicted bubble
     (predicted bbox + padding + masking, same treatment) to
     docs/phase3_eval/crops/unmatched_pred_*.png, for visual triage of whether each is a
     real bubble GT missed or a detector false positive.

Read-only measurement / asset generation. No OCR. No code under src/ touched.

Usage:
    PYTHONPATH=src python scripts/render_gt_bubble_crops.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import networkx as nx
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import roi_filter, centroid_nms
from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer
from measure_bubble_bbox_ratio import _load_gt_instrumentation, _match, BUBBLE_CLS_IDS

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
PHASE3_CFG_PATH = REPO / "configs" / "phase3.yaml"
PHASE4_CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]
OUT_DIR = REPO / "docs" / "phase3_eval" / "crops"
GT_CSV_PATH = REPO / "docs" / "phase3_eval" / "tags_gt.csv"


def _load_all_gt_nodes(gml: Path) -> list[tuple[str, str, float, float, float, float]]:
    """Return (node_id, label, x1, y1, x2, y2) for every GT node with a bbox."""
    g = nx.read_graphml(str(gml))
    out = []
    for nid, attrs in g.nodes(data=True):
        xmin = attrs.get("xmin")
        if xmin is None:
            continue
        out.append((nid, attrs.get("label", ""), float(xmin), float(attrs["ymin"]),
                    float(attrs["xmax"]), float(attrs["ymax"])))
    return out


def _crop_and_mask(
    im: Image.Image,
    x1: float, y1: float, x2: float, y2: float,
    all_boxes: list[tuple[str, str, float, float, float, float]],
    self_id: str,
    pad: int,
    upscale: int,
) -> Image.Image:
    """Crop (bbox + pad), white-fill any OTHER box (excluding background/self) that
    intersects the padded region, upscale by `upscale` (LANCZOS)."""
    W, H = im.size
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    cx2, cy2 = min(W, int(x2) + pad), min(H, int(y2) + pad)
    crop = im.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    for nid, label, ox1, oy1, ox2, oy2 in all_boxes:
        if nid == self_id or label == "background":
            continue
        ix1, iy1 = max(ox1, cx1), max(oy1, cy1)
        ix2, iy2 = min(ox2, cx2), min(oy2, cy2)
        if ix1 < ix2 and iy1 < iy2:
            draw.rectangle([ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1], fill=(255, 255, 255))
    return crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(PHASE3_CFG_PATH) as f:
        p3cfg = yaml.safe_load(f)
    with open(PHASE4_CFG_PATH) as f:
        p4cfg = yaml.safe_load(f)
    weights = REPO / args.weights
    cache_dir = REPO / args.cache_dir
    pad = p3cfg["crop_padding_px"]
    upscale = p3cfg["upscale_factor"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    unmatched_report = []

    for sheet_id in SHEETS:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        im = Image.open(png)

        all_gt_boxes = _load_all_gt_nodes(gml)
        gt_instr = [b for b in all_gt_boxes if b[1] == "instrumentation"]

        # Reading order: top-to-bottom, then left-to-right (bucket rows by ~60px bands
        # so slightly-offset bubbles on the "same row" still sort together).
        def _row_key(b):
            _, _, x1, y1, x2, y2 = b
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            return (round(cy / 60), cx)
        gt_instr_sorted = sorted(gt_instr, key=_row_key)

        for nid, label, x1, y1, x2, y2 in gt_instr_sorted:
            crop = _crop_and_mask(im, x1, y1, x2, y2, all_gt_boxes, nid, pad, upscale)
            crop.save(OUT_DIR / f"sheet{sheet_id}_{nid}.png")
            csv_rows.append({
                "sheet_id": sheet_id, "gt_node_id": nid,
                "function": "", "loop_number": "", "raw_tag": "",
            })

        # --- Unmatched predicted bubbles ---
        bg_bboxes, img_w, img_h, _ = _load_bg_bboxes(gml)
        raw_preds = _load_or_infer(sheet_id, png, weights, args.conf, args.iou, args.device, cache_dir)
        preds_after_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=p4cfg["roi_border_margin_frac"],
        )
        preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=p4cfg["nms_centroid_frac"])
        bubble_preds = [pr for pr in preds_deduped if pr["cls_id"] in BUBBLE_CLS_IDS]
        matches = _match(bubble_preds, [(b[0], b[2], b[3], b[4], b[5]) for b in gt_instr])
        # Identity-based, NOT coordinate-based: two distinct predictions (e.g. different
        # bubble subtypes) can share identical bbox coordinates, and a coordinate-tuple
        # key would wrongly mark BOTH as matched once either one matches.
        matched_pred_ids = {id(p) for p, _ in matches}

        # NOTE: no third-party masking here, deliberately. This is a one-off visual
        # triage of 14 unmatched predictions, and an early attempt at masking here hit a
        # real bug: masking against "every other predicted bubble" also masks a
        # near-duplicate/overlapping prediction sitting on the SAME physical bubble
        # (confirmed: sheet 0 has 1, sheet 10 has 2 such coordinate-duplicate pairs --
        # see the identity-based fix above), which blanked out genuine tag text. Simple
        # crop+pad+upscale, unmasked, is correct and sufficient for this triage step;
        # Task 2's production masking operates on GRAPH nodes post-dedup, not raw
        # per-class predictions, so it does not have this failure mode.
        idx = 0
        for pr in bubble_preds:
            if id(pr) in matched_pred_ids:
                continue
            idx += 1
            tag = f"unmatched_pred_sheet{sheet_id}_{idx}"
            x1, y1, x2, y2 = pr["x1"], pr["y1"], pr["x2"], pr["y2"]
            cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
            cx2, cy2 = min(im.width, int(x2) + pad), min(im.height, int(y2) + pad)
            crop = im.crop((cx1, cy1, cx2, cy2)).convert("RGB")
            crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
            crop.save(OUT_DIR / f"{tag}.png")
            unmatched_report.append({
                "sheet_id": sheet_id, "cls_name": pr["cls_name"], "conf": pr["conf"],
                "bbox": (x1, y1, x2, y2), "file": f"{tag}.png",
            })

    # --- Write CSV skeleton ---
    GT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sheet_id", "gt_node_id", "function", "loop_number", "raw_tag"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {len(csv_rows)} GT bubble crops -> {OUT_DIR}")
    print(f"Wrote CSV skeleton ({len(csv_rows)} rows) -> {GT_CSV_PATH}")

    print("\n" + "=" * 100)
    print(f"UNMATCHED PREDICTED BUBBLES ({len(unmatched_report)} total) -- for visual triage")
    print("=" * 100)
    print(f"{'sheet':<7}{'cls_name':<24}{'conf':<8}{'bbox':<32}{'file'}")
    for r in unmatched_report:
        print(f"{r['sheet_id']:<7}{r['cls_name']:<24}{r['conf']:<8.3f}{str(r['bbox']):<32}{r['file']}")


if __name__ == "__main__":
    main()
