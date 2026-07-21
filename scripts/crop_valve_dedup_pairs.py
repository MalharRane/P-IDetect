"""Phase 4 valve-family dedup: crop all 21 close cross-class valve-family pairs for
visual inspection BEFORE merging (docs/phase4_final.md node-dedupe correction, step 2).

Same measurement as scripts/measure_cross_class_duplicates.py, filtered to pairs where
BOTH sides are node_type=="valve" (different classes), with an annotated crop per pair
(both boxes drawn in different colours, union region + margin, 4x upscale) so each
class-pair group can be judged: one physical valve under two subtype guesses (merge)
vs two genuinely distinct drawn elements that still belong on one graph node
topologically (merge, but noted).

Read-only measurement / crop generation. No fix applied here.

Usage:
    PYTHONPATH=src python scripts/crop_valve_dedup_pairs.py --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import roi_filter, centroid_nms, _classify_pred
from run_phase4_steps03 import _load_bg_bboxes, _load_or_infer

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
SHEETS = [0, 3, 10]
OUT_DIR = REPO / "docs" / "phase4_step4_scope" / "valve_dedup_crops"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    weights = REPO / args.weights
    cache_dir = REPO / args.cache_dir
    centroid_frac = cfg["nms_centroid_frac"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_counts: dict = defaultdict(int)
    rows = []

    for sheet_id in SHEETS:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        im = Image.open(png)
        bg_bboxes, img_w, img_h, _ = _load_bg_bboxes(gml)
        raw_preds = _load_or_infer(sheet_id, png, weights, args.conf, args.iou, args.device, cache_dir)
        preds_after_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h, border_margin_frac=cfg["roi_border_margin_frac"],
        )
        preds_deduped, _ = centroid_nms(preds_after_roi, centroid_frac=centroid_frac)

        n = len(preds_deduped)
        for i in range(n):
            pi = preds_deduped[i]
            if _classify_pred(pi["cls_id"]) != "valve":
                continue
            wi, hi = pi["x2"] - pi["x1"], pi["y2"] - pi["y1"]
            cxi, cyi = (pi["x1"] + pi["x2"]) / 2, (pi["y1"] + pi["y2"]) / 2
            thr_i = centroid_frac * math.sqrt(max(wi * hi, 1.0))
            for j in range(i + 1, n):
                pj = preds_deduped[j]
                if pi["cls_id"] == pj["cls_id"]:
                    continue
                if _classify_pred(pj["cls_id"]) != "valve":
                    continue
                wj, hj = pj["x2"] - pj["x1"], pj["y2"] - pj["y1"]
                cxj, cyj = (pj["x1"] + pj["x2"]) / 2, (pj["y1"] + pj["y2"]) / 2
                thr_j = centroid_frac * math.sqrt(max(wj * hj, 1.0))
                d = math.hypot(cxi - cxj, cyi - cyj)
                thr = max(thr_i, thr_j)
                if d >= thr:
                    continue

                key = tuple(sorted((pi["cls_name"], pj["cls_name"])))
                group_counts[key] += 1
                idx = group_counts[key]
                tag = f"{key[0]}__{key[1]}__sheet{sheet_id}_{idx}"

                x1 = min(pi["x1"], pj["x1"]); y1 = min(pi["y1"], pj["y1"])
                x2 = max(pi["x2"], pj["x2"]); y2 = max(pi["y2"], pj["y2"])
                pad = 25
                crop = im.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad)).convert("RGB")
                crop = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
                draw = ImageDraw.Draw(crop)
                ox, oy = x1 - pad, y1 - pad
                draw.rectangle(
                    [(pi["x1"] - ox) * 4, (pi["y1"] - oy) * 4, (pi["x2"] - ox) * 4, (pi["y2"] - oy) * 4],
                    outline=(220, 0, 0), width=3,
                )
                draw.rectangle(
                    [(pj["x1"] - ox) * 4, (pj["y1"] - oy) * 4, (pj["x2"] - ox) * 4, (pj["y2"] - oy) * 4],
                    outline=(0, 100, 220), width=3,
                )
                fname = f"{tag}.png"
                crop.save(OUT_DIR / fname)
                rows.append((sheet_id, key, pi["cls_name"], pi["conf"], pj["cls_name"], pj["conf"], d, thr, fname))

    print(f"{'sheet':<7}{'class_pair':<55}{'A(red)':<22}{'confA':<8}{'B(blue)':<22}{'confB':<8}{'d':<7}{'thr':<7}{'file'}")
    for sheet_id, key, cls_a, conf_a, cls_b, conf_b, d, thr, fname in rows:
        pair_str = f"{key[0]} <-> {key[1]}"
        print(f"{sheet_id:<7}{pair_str:<55}{cls_a:<22}{conf_a:<8.3f}{cls_b:<22}{conf_b:<8.3f}{d:<7.1f}{thr:<7.1f}{fname}")

    print(f"\nTotal valve-family close pairs cropped: {len(rows)} -> {OUT_DIR}")
    print("\nBy class-pair:")
    for key, count in sorted(group_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key[0]} <-> {key[1]}: {count}")


if __name__ == "__main__":
    main()
