"""Fresh-upload check for the P3c direction-aware contraction fix (Fix 1).

Runs the production-shaped pipeline (no GT background bboxes, no GT vessel
injection -- mirrors src/pidetect/pipeline.py's "real upload" path, not the
OPEN100-eval-only helpers) on sheet 5, which is NOT one of the sheets used for
Fix 1's precision/recall measurement (0/3/10). Renders the contracted graph's
sym-to-sym edges twice -- once with the OLD all-pairs connector contraction,
once with the NEW direction-aware fix -- on the same downscaled sheet so the
before/after can be compared visually: real connections should still be drawn,
and any spurious cross-links that used to appear near multi-way junctions
should be reduced.

Usage:
    PYTHONPATH=src python scripts/fresh_upload_check_sheet5.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pidetect.graph.erase import (
    assert_no_duplicate_scored_nodes, scored_family_group_key,
    build_node_set, centroid_nms, erase_symbols, roi_filter,
)
from pidetect.graph.lines import binarize as _binarize_orig, run_step3, assert_run_step3_wired
from pidetect.graph.junction import run_step4
from pidetect.graph.build import run_step5, run_step6, build_sr_sc, veto_short_gap_by_crossing_separation

from run_phase4_steps03 import _run_inference
from measure_p3c_direction_fix import _run_step6_old

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
CACHE_DIR = REPO / "data" / "realworld_eval" / "open100" / "predictions"
OUT_DIR = REPO / "docs" / "phase4_step4_scope"
SHEET_ID = 5


def _load_or_infer(sheet_id: int, png: Path, weights: Path, conf: float, iou: float, device: str) -> list[dict]:
    cache_file = CACHE_DIR / f"{sheet_id}_preds.json"
    if cache_file.exists():
        print(f"  [cache] loading {cache_file.name}")
        return json.loads(cache_file.read_text())["predictions"]
    print("  running SAHI inference (slice=320, imgsz=640) ...")
    preds = _run_inference(png, weights, conf, iou, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"sheet": sheet_id, "predictions": preds}, indent=2))
    print(f"  [cache] saved -> {cache_file.name}  ({len(preds)} detections)")
    return preds


def _draw_overlay(png: Path, all_nodes, graph, out_path: Path, title: str, scale: float = 0.5) -> None:
    with Image.open(png) as im:
        img = im.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    node_by_gid = {}
    for n in all_nodes:
        node_by_gid[f"sym_{n.node_id}"] = n

    for u, v, edata in graph.edges(data=True):
        du, dv = graph.nodes.get(u), graph.nodes.get(v)
        if du is None or dv is None:
            continue
        ux, uy = du.get("cx"), du.get("cy")
        vx, vy = dv.get("cx"), dv.get("cy")
        if ux is None or vx is None:
            continue
        color = (255, 0, 0) if edata.get("contracted") and edata.get("via_type") == "connector" else (0, 160, 0)
        width = 3 if edata.get("contracted") else 2
        draw.line([(ux, uy), (vx, vy)], fill=color, width=width)

    for n, d in graph.nodes(data=True):
        if d.get("cx") is None:
            continue
        cx, cy = d["cx"], d["cy"]
        r = 6
        col = (60, 120, 220) if d.get("node_type") == "valve" else (
            (50, 180, 80) if d.get("node_type") == "instrument" else (150, 150, 150))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)

    new_w, new_h = int(W * scale), int(H * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    legend_h = 26
    legend = Image.new("RGB", (new_w, legend_h), (20, 20, 20))
    ld = ImageDraw.Draw(legend)
    ld.text((8, 5), f"{title}  |  red=contracted-via-connector  green=direct/other  circles=nodes", fill=(255, 255, 255))
    combined = Image.new("RGB", (new_w, new_h + legend_h), "white")
    combined.paste(img, (0, 0))
    combined.paste(legend, (0, new_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(out_path), "JPEG", quality=90)
    print(f"  -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    weights = REPO / args.weights
    png = RAW_DIR / f"{SHEET_ID}.png"

    raw_preds = _load_or_infer(SHEET_ID, png, weights, args.conf, args.iou, args.device)
    with Image.open(png) as im:
        img_w, img_h = im.size

    # Fresh-upload shape: no GT background bboxes, no GT vessel injection
    # (mirrors src/pidetect/pipeline.py, NOT the OPEN100-eval helper).
    preds_after_roi, _ = roi_filter(
        raw_preds, [], img_w, img_h,
        border_margin_frac=cfg["roi_border_margin_frac"],
    )
    preds_deduped, _ = centroid_nms(
        preds_after_roi, centroid_frac=cfg["nms_centroid_frac"],
        group_key=scored_family_group_key,
    )
    nodes = build_node_set(preds_deduped)
    assert_no_duplicate_scored_nodes(nodes, centroid_frac=cfg["nms_centroid_frac"])
    print(f"  Nodes: {len(nodes)}  (raw preds: {len(raw_preds)}, after ROI: {len(preds_after_roi)})")

    img_bgr = cv2.imread(str(png))
    _, binary_pre_erase = _binarize_orig(img_bgr, blur_sigma=cfg["blur_sigma"], close_disk_r=0, open_disk_r=0)
    erased_bgr = erase_symbols(img_bgr, nodes, dilation_px=cfg["erase_dilation_px"])

    step3_kwargs = dict(
        blur_sigma=cfg["blur_sigma"],
        close_disk_r=cfg["morph_close_disk"],
        open_disk_r=cfg["morph_open_disk"],
        min_branch_len_px=cfg["min_branch_len_px"],
        endpoint_bind_extra_px=cfg["endpoint_bind_extra_px"],
        off_page_border_px=cfg["off_page_border_px"],
        mask_all_nodes=cfg["mask_all_nodes"],
        mask_all_nodes_fallback=cfg["mask_all_nodes_fallback"],
        short_gap_max_px=cfg["short_gap_max_px"],
        interpose_tol_px=cfg["short_gap_interpose_tol_px"],
        corridor_half_width_px=cfg["short_gap_corridor_half_width_px"],
        continuity_frac=cfg["short_gap_continuity_frac"],
        perp_reject_ratio=cfg["short_gap_perp_reject_ratio"],
        stub_angle_tol_deg=cfg["short_gap_stub_angle_tol_deg"],
    )
    assert_run_step3_wired(cfg, step3_kwargs)
    s3 = run_step3(erased_bgr, nodes, img_w, img_h, binary_pre_erase=binary_pre_erase, **step3_kwargs)
    s4 = run_step4(
        s3.skeleton, s3.branches, s3.endpoints,
        collinear_angle_tol=cfg["collinear_angle_tol"],
        min_crossing_gap=cfg["min_crossing_gap"],
        max_crossing_gap=cfg["max_crossing_gap"],
        elbow_angle_min=cfg["elbow_angle_min"],
        junction_radius_R=cfg["junction_radius_R"],
        binary_raw=s3.binary_raw,
        exclude_rects=[],
    )

    collinear_tol_deg = cfg["contraction_collinear_tol_deg"]
    SR, Sc = build_sr_sc(nodes, s3.off_page_nodes, s3, s4, collinear_tol_deg)
    kept_pairs, suppressed = veto_short_gap_by_crossing_separation(s3.short_gap_pairs, SR, Sc)
    import dataclasses
    s3_vetoed = dataclasses.replace(s3, short_gap_pairs=kept_pairs)

    s5 = run_step5(nodes, s3.off_page_nodes, s3_vetoed, s4)
    s6_old = _run_step6_old(s5)
    s6_new = run_step6(s5, collinear_tol_deg)

    print(f"  Step5 (raw) nodes={s5.n_nodes} edges={s5.n_edges}")
    print(f"  Step6 OLD (all-pairs):      nodes={s6_old.n_nodes} edges={s6_old.n_edges} "
          f"connectors_contracted={s6_old.n_contracted_connectors}")
    print(f"  Step6 NEW (direction-aware): nodes={s6_new.n_nodes} edges={s6_new.n_edges} "
          f"connectors_contracted={s6_new.n_contracted_connectors}")

    old_edges = {frozenset((u, v)) for u, v in s6_old.graph.edges()}
    new_edges = {frozenset((u, v)) for u, v in s6_new.graph.edges()}
    removed = old_edges - new_edges
    added = new_edges - old_edges
    print(f"  Edges removed by fix: {len(removed)}  {sorted(tuple(e) for e in removed)}")
    print(f"  Edges added by fix:   {len(added)}  {sorted(tuple(e) for e in added)}")

    all_nodes_for_draw = nodes + s3.off_page_nodes
    _draw_overlay(png, all_nodes_for_draw, s6_old.graph, OUT_DIR / "sheet5_freshupload_OLD_allpairs.jpg",
                  "Sheet 5 -- OLD (all-pairs contraction)")
    _draw_overlay(png, all_nodes_for_draw, s6_new.graph, OUT_DIR / "sheet5_freshupload_NEW_directionaware.jpg",
                  "Sheet 5 -- NEW (direction-aware contraction, Fix 1)")


if __name__ == "__main__":
    main()
