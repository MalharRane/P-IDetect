"""Unbound-tip audit for Phase 4 step-3 binding.

For each unbound, non-off-page endpoint from a free-tip branch (type 0 or 1),
classifies it into one of three buckets:

  (a) No detected symbol within 2× extra_bind_px of the endpoint
      -> legitimately dangling (broken line, interior junction, etc.)
  (b) A symbol is within 2× but outside 1× bind zone, AND that symbol has
      no other endpoints bound to it yet
      -> binding-radius miss (would connect if we expanded the radius)
  (c) A symbol is within 2× but outside 1× bind zone, AND that symbol already
      has other endpoints bound to it
      -> near-miss where the symbol is already reachable via another stub
         (expanding radius would add a parallel edge, not recover a lost one)

Vessel detection is disabled (vessel_max_loop_area_px2=0) so the audit
reflects the pre-fix binding state.

Outputs:
  docs/phase4_step0_3/unbound_audit_table.txt   — per-sheet bucket counts
  docs/phase4_step0_3/unbound_audit_crops.jpg   — 6 example crops (b+c tips)

Usage:
    PYTHONPATH=src python scripts/audit_unbound_tips.py \\
        --weights runs/detect/train_small_objects/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.graph.erase import (
    SymbolNode,
    build_node_set,
    centroid_nms,
    erase_symbols,
    roi_filter,
)
from pidetect.graph.lines import BoundEndpoint, Branch, Step3Result, run_step3

RAW_DIR  = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
OUT_DIR  = REPO / "docs" / "phase4_step0_3"
SHEETS   = [0, 3, 10]


# ---------------------------------------------------------------------------
# Helpers shared with run_phase4_steps03
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_bg_bboxes(gml: Path) -> tuple[list, int, int]:
    import networkx as nx
    from PIL import Image

    g = nx.read_graphml(str(gml))
    bg_bboxes = []
    for _, attrs in g.nodes(data=True):
        if attrs.get("label") == "background" and "xmin" in attrs:
            bg_bboxes.append((
                float(attrs["xmin"]), float(attrs["ymin"]),
                float(attrs["xmax"]), float(attrs["ymax"]),
            ))
    png = gml.with_suffix(".png")
    with Image.open(png) as im:
        img_w, img_h = im.size
    return bg_bboxes, img_w, img_h


def _load_preds(sheet_id: int, cache_dir: Path) -> list[dict]:
    cache_file = cache_dir / f"{sheet_id}_preds.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            f"Run run_phase4_steps03.py first to generate predictions."
        )
    return json.loads(cache_file.read_text())["predictions"]


def _in_dilated_box(px: int, py: int,
                    dx1: int, dy1: int, dx2: int, dy2: int, extra: int) -> bool:
    return (dx1 - extra <= px <= dx2 + extra and
            dy1 - extra <= py <= dy2 + extra)


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------

def bucket_unbound_tips(
    branches: list[Branch],
    endpoints: list[BoundEndpoint],
    nodes: list[SymbolNode],
    extra_bind_px: int,
) -> tuple[list, list, list]:
    """Classify unbound, non-off-page, free-tip endpoints into buckets a/b/c.

    Returns (bucket_a, bucket_b, bucket_c) where each entry is:
      bucket_a: (ep,)                          — no nearby symbol
      bucket_b: (ep, nearby_node, dist_px)     — radius miss, symbol unbound
      bucket_c: (ep, nearby_node, dist_px)     — near-miss, symbol already has ports
    """
    branch_map = {b.branch_id: b for b in branches}
    graph_nodes = [n for n in nodes if n.is_graph_node]

    bucket_a: list = []
    bucket_b: list = []
    bucket_c: list = []

    for ep in endpoints:
        if ep.bound_node_id is not None:
            continue
        if ep.is_off_page:
            continue

        b = branch_map.get(ep.branch_id)
        if b is None or b.branch_type not in (0, 1):
            continue  # skip junction-junction and cycle endpoints

        px, py = ep.point_xy

        # Collect symbols within 2× radius
        within_2x: list[tuple[float, SymbolNode]] = []
        for node in graph_nodes:
            dx1, dy1, dx2, dy2 = node.dilated
            if _in_dilated_box(px, py, dx1, dy1, dx2, dy2, 2 * extra_bind_px):
                dist = math.hypot(px - node.cx, py - node.cy)
                within_2x.append((dist, node))

        if not within_2x:
            bucket_a.append((ep,))
        else:
            # Nearest symbol within 2×
            within_2x.sort(key=lambda t: t[0])
            dist, best = within_2x[0]
            if best.ports:
                bucket_c.append((ep, best, dist))
            else:
                bucket_b.append((ep, best, dist))

    return bucket_a, bucket_b, bucket_c


# ---------------------------------------------------------------------------
# Crop rendering
# ---------------------------------------------------------------------------

_SKEL_COLOR_BGR = (60, 60, 255)   # red in BGR
_EP_COLOR_BGR   = (0, 0, 220)     # red circle
_SYM_COLOR_BGR  = (0, 220, 220)   # yellow rect (BGR)
_BIND_EDGE_BGR  = (120, 255, 120) # green — 1x bind zone edge
_2X_EDGE_BGR    = (200, 200, 40)  # cyan — 2x bind zone edge


def _render_crop(
    img_bgr: np.ndarray,
    skeleton: np.ndarray,
    ep: BoundEndpoint,
    nearby_node: SymbolNode | None,
    extra_bind_px: int,
    crop_px: int = 160,
    label: str = "",
) -> np.ndarray:
    """Return a crop_px × crop_px BGR image centred on the endpoint."""
    H, W = img_bgr.shape[:2]
    px, py = ep.point_xy
    half = crop_px // 2

    x1 = max(0, px - half); y1 = max(0, py - half)
    x2 = min(W, x1 + crop_px); y2 = min(H, y1 + crop_px)
    x1 = max(0, x2 - crop_px); y1 = max(0, y2 - crop_px)

    crop = img_bgr[y1:y2, x1:x2].copy()
    sk   = skeleton[y1:y2, x1:x2]
    crop[sk] = _SKEL_COLOR_BGR

    # Endpoint marker
    ex, ey = px - x1, py - y1
    cv2.circle(crop, (ex, ey), 5, _EP_COLOR_BGR, 2)
    cv2.circle(crop, (ex, ey), 2, _EP_COLOR_BGR, -1)

    if nearby_node is not None:
        dx1, dy1, dx2, dy2 = nearby_node.dilated
        # 1× bind zone
        cv2.rectangle(
            crop,
            (max(0, dx1 - extra_bind_px - x1), max(0, dy1 - extra_bind_px - y1)),
            (min(crop_px - 1, dx2 + extra_bind_px - x1), min(crop_px - 1, dy2 + extra_bind_px - y1)),
            _BIND_EDGE_BGR, 1,
        )
        # 2× bind zone
        cv2.rectangle(
            crop,
            (max(0, dx1 - 2*extra_bind_px - x1), max(0, dy1 - 2*extra_bind_px - y1)),
            (min(crop_px - 1, dx2 + 2*extra_bind_px - x1), min(crop_px - 1, dy2 + 2*extra_bind_px - y1)),
            _2X_EDGE_BGR, 1,
        )
        # Symbol bbox itself
        cv2.rectangle(
            crop,
            (max(0, int(nearby_node.x1) - x1), max(0, int(nearby_node.y1) - y1)),
            (min(crop_px - 1, int(nearby_node.x2) - x1), min(crop_px - 1, int(nearby_node.y2) - y1)),
            _SYM_COLOR_BGR, 2,
        )

    # Label strip at top
    if label:
        cv2.rectangle(crop, (0, 0), (crop_px - 1, 14), (30, 30, 30), -1)
        cv2.putText(crop, label, (3, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    # Pad if edge crop is smaller than crop_px
    ph = crop_px - crop.shape[0]
    pw = crop_px - crop.shape[1]
    if ph > 0 or pw > 0:
        crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw,
                                  cv2.BORDER_CONSTANT, value=(220, 220, 220))
    return crop


def render_audit_montage(
    sheet_examples: list[tuple],  # (sheet_id, img_bgr, skel, ep, node|None, dist|None, label)
    crop_px: int = 160,
    n_cols: int = 3,
    out_path: Path | None = None,
) -> np.ndarray:
    """Arrange up to 6 example crops in a 2×3 grid and optionally save."""
    panels = []
    extra_bind_px = 5  # used for bind zone drawing only

    for entry in sheet_examples[:6]:
        sheet_id, img_bgr, skel, ep, node, dist, label = entry
        panels.append(
            _render_crop(img_bgr, skel, ep, node, extra_bind_px, crop_px, label)
        )

    # Pad to 6 panels
    blank = np.full((crop_px, crop_px, 3), 210, dtype=np.uint8)
    while len(panels) < 6:
        panels.append(blank.copy())

    row0 = np.hstack(panels[:3])
    row1 = np.hstack(panels[3:6])
    montage = np.vstack([row0, row1])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return montage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 unbound-tip audit")
    p.add_argument("--weights", default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--cache-dir", default="data/realworld_eval/open100/predictions")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.5)
    p.add_argument("--device", default="")
    p.add_argument("--sheets", nargs="+", type=int, default=SHEETS)
    args = p.parse_args()

    cache_dir = REPO / args.cache_dir
    cfg = _load_cfg()
    extra_bind_px = cfg["endpoint_bind_extra_px"]

    print("=" * 60)
    print("Phase 4 -- Unbound-tip audit  (vessel detection DISABLED)")
    print("=" * 60)
    print(f"  1x bind zone: dilated_bbox + {extra_bind_px}px extra")
    print(f"  2x bind zone: dilated_bbox + {2*extra_bind_px}px extra")
    print()

    # ---- per-sheet header for table
    header = (
        f"  {'Sheet':<8} {'free_tip_unbd':>14} "
        f"{'(a) dangling':>14} {'(b) radius_miss':>16} {'(c) near_miss':>14}"
    )

    all_lines: list[str] = [header, "  " + "-" * 72]
    crop_examples: list[tuple] = []  # (b) and (c) examples for montage

    for sheet_id in args.sheets:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        if not (png.exists() and gml.exists()):
            print(f"[warn] sheet {sheet_id}: missing files, skipping")
            continue

        print(f"\n--- Sheet {sheet_id} ---")
        bg_bboxes, img_w, img_h = _load_bg_bboxes(gml)

        raw_preds = _load_preds(sheet_id, cache_dir)
        preds_roi, _ = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h,
            border_margin_frac=cfg["roi_border_margin_frac"],
        )
        preds_nms, _ = centroid_nms(preds_roi, centroid_frac=cfg["nms_centroid_frac"])
        nodes = build_node_set(preds_nms)

        img_bgr = cv2.imread(str(png))
        erased  = erase_symbols(img_bgr, nodes, dilation_px=cfg["erase_dilation_px"])

        print("  Running step 3 (vessel detection disabled) …")
        s3 = run_step3(
            erased, nodes, img_w, img_h,
            blur_sigma=cfg["blur_sigma"],
            close_disk_r=cfg["morph_close_disk"],
            open_disk_r=cfg["morph_open_disk"],
            min_branch_len_px=cfg["min_branch_len_px"],
            endpoint_bind_extra_px=extra_bind_px,
            off_page_border_px=cfg["off_page_border_px"],
        )

        all_nodes_for_audit = nodes + s3.off_page_nodes
        b_a, b_b, b_c = bucket_unbound_tips(
            s3.branches, s3.endpoints, all_nodes_for_audit, extra_bind_px
        )

        n_free_unbd = len(b_a) + len(b_b) + len(b_c)
        row = (
            f"  {sheet_id:<8} {n_free_unbd:>14} "
            f"{len(b_a):>14} {len(b_b):>16} {len(b_c):>14}"
        )
        all_lines.append(row)
        print(row.strip())

        # Collect crop examples (up to 3 from b, 3 from c, across all sheets)
        for ep, node, dist in b_b[:3]:
            if len(crop_examples) < 3:
                lbl = f"S{sheet_id} (b) d={dist:.0f}px {node.cls_name[:12]}"
                crop_examples.append((sheet_id, img_bgr, s3.skeleton, ep, node, dist, lbl))

        for ep, node, dist in b_c[:3]:
            if len(crop_examples) < 6:
                lbl = f"S{sheet_id} (c) d={dist:.0f}px {node.cls_name[:12]}"
                crop_examples.append((sheet_id, img_bgr, s3.skeleton, ep, node, dist, lbl))

    all_lines.append("")
    all_lines.append("  Legend:")
    all_lines.append("    (a) No symbol within 2x bind zone -- legitimately dangling")
    all_lines.append("    (b) Symbol within 2x but outside 1x, symbol has NO other bound ports")
    all_lines.append("         -> would recover a connection by expanding radius")
    all_lines.append("    (c) Symbol within 2x but outside 1x, symbol already has bound ports")
    all_lines.append("         -> expanding radius would add parallel edge only")
    all_lines.append("")
    all_lines.append("  Crop legend (in the 160x160px images):")
    all_lines.append("    red circle    = unbound endpoint")
    all_lines.append("    yellow rect   = nearby symbol bbox")
    all_lines.append("    green border  = current 1x bind zone  (dilated_bbox + 5px)")
    all_lines.append("    cyan border   = 2x bind zone  (dilated_bbox + 10px)")

    table_str = "\n".join(all_lines)
    print("\n" + table_str)

    table_out = OUT_DIR / "unbound_audit_table.txt"
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.write_text(table_str, encoding="utf-8")
    print(f"\nTable -> {table_out}")

    # Render montage
    crops_out = OUT_DIR / "unbound_audit_crops.jpg"
    if crop_examples:
        render_audit_montage(crop_examples, crop_px=160, out_path=crops_out)
        print(f"Crops -> {crops_out}  ({len(crop_examples)} panels)")
    else:
        print("No (b)/(c) examples found -- no crops generated.")

    print("\nDone.")


if __name__ == "__main__":
    main()
