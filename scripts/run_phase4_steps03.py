"""Phase 4 steps 0-4 diagnostic runner.

Runs preprocessing + node-set building + symbol erasure + skeleton extraction
+ endpoint binding + junction analysis on OPEN100 sheets 0, 3, 10.  Produces:

  docs/phase4_step0_3/sheet_{N}_diag.jpg   -- diagnostic render
  docs/phase4_step0_3/sheet_{N}_erased.jpg -- symbol-erased image (for inspection)
  docs/phase4_step0_3/stats.txt            -- per-sheet binding + junction statistics

Usage (requires weights):
    PYTHONPATH=src python scripts/run_phase4_steps03.py \\
        --weights runs/detect/train_small_objects/weights/best.pt

To re-use cached inference results from a previous run (skips SAHI):
    PYTHONPATH=src python scripts/run_phase4_steps03.py \\
        --weights runs/detect/train_small_objects/weights/best.pt \\
        --cache-dir data/realworld_eval/open100/predictions
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.data.open100 import parse_graphml
from pidetect.graph.erase import (
    SymbolNode,
    build_node_set,
    centroid_nms,
    erase_symbols,
    roi_filter,
)
from pidetect.graph.lines import (
    Step3Result, binarize as _binarize_orig, run_step3, assert_run_step3_wired,
)
from pidetect.graph.junction import Step4Result, run_step4
from pidetect.graph.build import (
    Step5Result, Step6Result, Step7Result,
    run_step5, run_step6, run_step7,
    build_sr_sc, veto_short_gap_by_crossing_separation,
)
from pidetect.graph.export import run_step8
from pidetect.graph.evaluate import Step9Result, run_step9

RAW_DIR  = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
OUT_DIR  = REPO / "docs" / "phase4_step0_3"
SHEETS   = [0, 3, 10]

# Implausible port count thresholds per node type (for diagnostic flagging)
_PORT_LIMITS: dict[str, tuple[int, int]] = {
    "valve":           (0, 4),   # most valves have 2 ports; allow 0-4
    "instrument":      (0, 4),
    "unknown_fitting": (0, 6),   # reducers/tees can have 3
    "off_page":        (0, 2),
    "vessel":          (0, 15),  # large equipment -- many nozzles possible
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Inference (SAHI) + caching
# ---------------------------------------------------------------------------

def _run_inference(
    png: Path, weights: Path,
    conf: float, iou: float, device: str,
    slice_px: int = 320, imgsz: int = 640, overlap: float = 0.2,
) -> list[dict]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights),
        confidence_threshold=conf,
        device=device,
    )
    result = get_sliced_prediction(
        image=str(png),
        detection_model=model,
        slice_height=slice_px,
        slice_width=slice_px,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=False,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=iou,
        auto_slice_resolution=False,
        verbose=0,
    )
    preds = []
    for obj in result.object_prediction_list:
        xyxy = obj.bbox.to_xyxy()
        preds.append({
            "cls_id":   obj.category.id,
            "cls_name": obj.category.name,
            "conf":     round(float(obj.score.value), 4),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return preds


def _load_or_infer(
    sheet_id: int, png: Path, weights: Path,
    conf: float, iou: float, device: str, cache_dir: Path,
) -> list[dict]:
    cache_file = cache_dir / f"{sheet_id}_preds.json"
    if cache_file.exists():
        print(f"  [cache] loading {cache_file.name}")
        data = json.loads(cache_file.read_text())
        return data["predictions"]

    print(f"  running SAHI inference (slice=320, imgsz=640) ...")
    preds = _run_inference(png, weights, conf, iou, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"sheet": sheet_id, "predictions": preds}, indent=2))
    print(f"  [cache] saved -> {cache_file.name}  ({len(preds)} detections)")
    return preds


# ---------------------------------------------------------------------------
# GT background bbox loader
# ---------------------------------------------------------------------------

_GT_VESSEL_LABELS: frozenset[str] = frozenset({"tank", "pump"})


def _load_bg_bboxes(
    gml: Path,
) -> tuple[list[tuple[float, float, float, float]], int, int,
           list[tuple[str, float, float, float, float]]]:
    """Parse GraphML for background bboxes, image size, and GT vessel bboxes.

    Returns (bg_bboxes, img_w, img_h, vessel_entries) where each vessel_entry
    is (label, x1, y1, x2, y2) for nodes with label in {tank, pump}.
    """
    import networkx as nx

    g = nx.read_graphml(str(gml))
    bg_bboxes: list[tuple[float, float, float, float]] = []
    vessel_entries: list[tuple[str, float, float, float, float]] = []
    max_x, max_y = 0.0, 0.0

    for _, attrs in g.nodes(data=True):
        label = attrs.get("label", "")
        xmin = attrs.get("xmin")
        if xmin is None:
            continue
        x1 = float(xmin); y1 = float(attrs["ymin"])
        x2 = float(attrs["xmax"]); y2 = float(attrs["ymax"])
        max_x = max(max_x, x2); max_y = max(max_y, y2)
        if label == "background":
            bg_bboxes.append((x1, y1, x2, y2))
        elif label in _GT_VESSEL_LABELS:
            vessel_entries.append((label, x1, y1, x2, y2))

    png = gml.with_suffix(".png")
    if png.exists():
        with Image.open(png) as im:
            img_w, img_h = im.size
    else:
        img_w, img_h = int(max_x) + 1, int(max_y) + 1

    return bg_bboxes, img_w, img_h, vessel_entries


def _build_vessel_nodes(
    vessel_entries: list[tuple[str, float, float, float, float]],
    start_id: int,
) -> list[SymbolNode]:
    """Create SymbolNode objects for GT-annotated tank/pump vessel bodies."""
    return [
        SymbolNode(
            node_id=start_id + i,
            cls_id=-2,
            cls_name=label,
            node_type="vessel",
            x1=x1, y1=y1, x2=x2, y2=y2,
            conf=1.0,
        )
        for i, (label, x1, y1, x2, y2) in enumerate(vessel_entries)
    ]


# ---------------------------------------------------------------------------
# Diagnostic render
# ---------------------------------------------------------------------------

# Colours (RGB) per node type
_NODE_COLORS: dict[str, tuple[int, int, int]] = {
    "valve":           (60,  120, 220),   # blue
    "instrument":      (50,  180,  80),   # green
    "unknown_fitting": (220, 130,  30),   # orange
    "flow_arrow":      (180,  50, 180),   # purple
    "tag_rect":        (160, 160, 160),   # grey
    "off_page":        (220,  30,  30),   # red
    "vessel":          (0,   190, 180),   # teal -- synthetic vessel body nodes
}
_SKEL_COLOR       = (255,  60,  60)    # red skeleton
_BIND_COLOR       = (0,   210, 210)    # cyan binding lines
_CONTEST_COLOR    = (255, 255,   0)    # yellow contested endpoints
_CONNECTOR_COLOR  = (255,   0, 255)    # magenta connector dots
_CROSSING_COLOR   = (255, 165,   0)    # orange crossing marks


def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_diagnostic(
    raw_png: Path,
    all_nodes: list[SymbolNode],
    off_page_nodes: list[SymbolNode],
    result: Step3Result,
    out_path: Path,
    scale: float = 0.35,
    s4: Step4Result | None = None,
) -> None:
    """Draw nodes + skeleton + binding lines (+ junctions if s4 given) on a downscaled sheet."""
    with Image.open(raw_png) as im:
        img_rgb = im.convert("RGB")

    W, H = img_rgb.size

    # --- overlay skeleton in red ---
    skel_arr = np.array(img_rgb)
    skel_mask = result.skeleton
    # brighten background pixels slightly, paint skeleton red
    skel_arr[skel_mask] = _SKEL_COLOR
    img_rgb = Image.fromarray(skel_arr)

    draw = ImageDraw.Draw(img_rgb, "RGBA")

    # --- draw node bboxes ---
    for node in all_nodes + off_page_nodes:
        color = _NODE_COLORS.get(node.node_type, (200, 200, 200))
        x1, y1, x2, y2 = int(node.x1), int(node.y1), int(node.x2), int(node.y2)
        lw = 3 if node.node_type in ("valve", "instrument") else 2
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=lw)

    # --- draw binding lines: endpoint -> symbol centroid ---
    bound_by_node: dict[int, list[tuple[int, int]]] = {}
    for ep in result.endpoints:
        if ep.bound_node_id is not None:
            bound_by_node.setdefault(ep.bound_node_id, []).append(ep.point_xy)

    all_nodes_map: dict[int, SymbolNode] = {
        n.node_id: n for n in all_nodes + off_page_nodes
    }
    for nid, pts in bound_by_node.items():
        node = all_nodes_map.get(nid)
        if node is None:
            continue
        ncx, ncy = int(node.cx), int(node.cy)
        color = _BIND_COLOR + (200,)
        for px, py in pts:
            draw.line([(px, py), (ncx, ncy)], fill=color, width=1)

    # --- mark contested endpoints in yellow ---
    for ep in result.endpoints:
        if ep.contested:
            px, py = ep.point_xy
            draw.ellipse(
                [px - 4, py - 4, px + 4, py + 4],
                outline=_CONTEST_COLOR + (255,),
                width=2,
            )

    # --- junction overlays (step 4) ---
    if s4 is not None:
        for jn in s4.connectors:
            cx, cy = int(jn.cx), int(jn.cy)
            r = 6
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=_CONNECTOR_COLOR + (255,),
                width=2,
            )
        for jn in s4.crossings:
            cx, cy = int(jn.cx), int(jn.cy)
            r = 7
            # X mark
            draw.line([(cx - r, cy - r), (cx + r, cy + r)],
                      fill=_CROSSING_COLOR + (255,), width=2)
            draw.line([(cx + r, cy - r), (cx - r, cy + r)],
                      fill=_CROSSING_COLOR + (255,), width=2)

    # --- downscale ---
    new_w, new_h = int(W * scale), int(H * scale)
    img_rgb = img_rgb.resize((new_w, new_h), Image.LANCZOS)

    # --- legend strip ---
    font = _try_font(14)
    legend_h = 24
    legend = Image.new("RGB", (new_w, legend_h), (20, 20, 20))
    ld = ImageDraw.Draw(legend)
    entries = [
        ("valve", _NODE_COLORS["valve"]),
        ("instrument", _NODE_COLORS["instrument"]),
        ("unknown_fitting", _NODE_COLORS["unknown_fitting"]),
        ("off_page", _NODE_COLORS["off_page"]),
        ("skeleton", _SKEL_COLOR),
        ("binding", _BIND_COLOR),
        ("connector", _CONNECTOR_COLOR),
        ("crossing", _CROSSING_COLOR),
    ]
    x = 6
    for label, color in entries:
        ld.text((x, 5), f"■ {label}", fill=color, font=font)
        x += len(label) * 8 + 24

    combined = Image.new("RGB", (new_w, new_h + legend_h), "white")
    combined.paste(img_rgb, (0, 0))
    combined.paste(legend, (0, new_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(out_path), "JPEG", quality=88)
    print(f"  Diagnostic render -> {out_path.name}  ({new_w}x{new_h + legend_h}px)")


# ---------------------------------------------------------------------------
# Stats + report
# ---------------------------------------------------------------------------

def _port_stats(all_nodes: list[SymbolNode], off_page_nodes: list[SymbolNode]) -> dict:
    by_type: Counter = Counter()
    implausible: list[str] = []
    for node in all_nodes + off_page_nodes:
        if not node.is_graph_node:
            continue
        by_type[node.node_type] += 1
        n_ports = len(node.ports)
        lo, hi = _PORT_LIMITS.get(node.node_type, (0, 99))
        if not (lo <= n_ports <= hi):
            implausible.append(
                f"    {node.cls_name} (id={node.node_id}): {n_ports} ports"
                f"  [expected {lo}-{hi}]"
            )
    return {"by_type": dict(by_type), "implausible": implausible}


def print_sheet_stats(
    sheet_id: int,
    raw_preds: int,
    after_roi: int,
    after_nms: int,
    all_nodes: list[SymbolNode],
    off_page_nodes: list[SymbolNode],
    result: Step3Result,
    s4: Step4Result | None = None,
    s5: Step5Result | None = None,
    s6: Step6Result | None = None,
    s7: Step7Result | None = None,
    s9: Step9Result | None = None,
) -> list[str]:
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)
        print(s)

    ln(f"\n{'='*58}")
    ln(f"  Sheet {sheet_id}")
    ln(f"{'='*58}")
    ln(f"  Detections -- raw: {raw_preds}, after ROI: {after_roi}, after NMS: {after_nms}")
    ln(f"  Detections suppressed by ROI:  {raw_preds - after_roi}")
    ln(f"  Detections suppressed by NMS:  {after_roi - after_nms}")
    ln()

    stats = _port_stats(all_nodes, off_page_nodes)
    ln("  Nodes by type:")
    type_order = ["valve", "instrument", "unknown_fitting", "vessel", "off_page"]
    for t in type_order:
        n = stats["by_type"].get(t, 0)
        ln(f"    {t:<20} {n}")
    other_types = [k for k in stats["by_type"] if k not in type_order]
    for t in other_types:
        ln(f"    {t:<20} {stats['by_type'][t]}")
    total_graph_nodes = sum(stats["by_type"].values())
    ln(f"    {'TOTAL (graph nodes)':<20} {total_graph_nodes}")
    flow_arrows = sum(1 for n in all_nodes if n.node_type == "flow_arrow")
    tag_rects = sum(1 for n in all_nodes if n.node_type == "tag_rect")
    if flow_arrows:
        ln(f"    flow_arrow (set aside) {flow_arrows}")
    if tag_rects:
        ln(f"    tag_rect (erased only)  {tag_rects}")
    ln()

    ln(f"  Skeleton branches:     {len(result.branches)}")
    ln(f"  Endpoints total:       {len(result.endpoints)}")
    ln(f"  Endpoints bound:       {result.n_bound}")
    ln(f"  Endpoints unbound:     {result.n_unbound}")
    ln(f"  Stub contests:         {result.n_contests}")
    ln(f"  Off-page nodes created:{result.n_off_page}")
    if result.short_gap_pairs:
        ln(f"  Short-gap pairs (pre-erasure path confirmed): {len(result.short_gap_pairs)}")
    ln()

    if stats["implausible"]:
        ln(f"  Implausible port counts ({len(stats['implausible'])} symbols):")
        for s in stats["implausible"]:
            ln(s)
    else:
        ln("  No symbols with implausible port counts.")

    if s4 is not None:
        ln()
        subtypes = s4.connector_subtypes()
        ln(f"  Connectors detected:   {s4.n_connectors}")
        for sub in ("elbow", "tee", "4way", "nway", "straight", "stub"):
            n = subtypes.get(sub, 0)
            if n:
                ln(f"    {sub:<12} {n}")
        ln(f"  Crossings detected:    {s4.n_crossings}")

    if s5 is not None:
        ln()
        ln("  Graph (step 5 -- raw, pre-contraction):")
        type_counts = s5.node_type_counts()
        ln(f"    Nodes total:         {s5.n_nodes}")
        for t in ("valve", "instrument", "unknown_fitting", "vessel", "off_page",
                  "connector", "crossing"):
            c = type_counts.get(t, 0)
            if c:
                ln(f"      {t:<20} {c}")
        ln(f"    Edges total:         {s5.n_edges}")
        ln(f"    Dangling branches:   {s5.n_dangling}")
        cs = s5.components_summary()
        ln(f"    Components:          {cs['n_components']}"
           f"  (largest={cs['largest']}, isolated={cs['isolated']})")

    if s6 is not None:
        ln()
        ln("  Graph (step 6 -- contracted, symbol-to-symbol):")
        type_counts = s6.node_type_counts()
        ln(f"    Contracted connectors: {s6.n_contracted_connectors}")
        ln(f"    Contracted crossings:  {s6.n_contracted_crossings}")
        ln(f"    Nodes remaining:     {s6.n_nodes}")
        for t in ("valve", "instrument", "unknown_fitting", "vessel", "off_page"):
            c = type_counts.get(t, 0)
            if c:
                ln(f"      {t:<20} {c}")
        ln(f"    Edges total:         {s6.n_edges}")
        cs = s6.components_summary()
        ln(f"    Components:          {cs['n_components']}"
           f"  (largest={cs['largest']}, isolated={cs['isolated']})")

    if s7 is not None:
        ln()
        ln("  Arrows (step 7 -- direction tagging):")
        ln(f"    Arrows total:        {s7.n_arrows_total}")
        assign_pct = 100 * s7.n_arrows_assigned / max(s7.n_arrows_total, 1)
        ln(f"    Assigned to edge:    {s7.n_arrows_assigned}"
           f"  ({assign_pct:.0f}%)")
        if s7.n_arrows_directed > 0:
            dir_pct = 100 * s7.n_arrows_directed / max(s7.n_arrows_assigned, 1)
            ln(f"    Direction tagged:    {s7.n_arrows_directed}"
               f"  ({dir_pct:.0f}% of assigned)")
        ln(f"    Edges directed:      {s7.n_edges_directed}"
           f" / {s6.n_edges if s6 else '?'}")

    if s9 is not None:
        ln()
        ln("  Evaluation vs GT (step 9):")
        for line in s9.summary_lines():
            ln(line)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 4 steps 0-3 diagnostic runner"
    )
    p.add_argument(
        "--weights",
        default="runs/detect/train_small_objects/weights/best.pt",
        help="Path to .pt weights",
    )
    p.add_argument(
        "--cache-dir",
        default="data/realworld_eval/open100/predictions",
        help="Directory for cached per-sheet inference JSONs",
    )
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.5)
    p.add_argument("--device", default="",
                   help="'' = auto, 'cpu', '0' = GPU 0")
    p.add_argument("--sheets", nargs="+", type=int, default=SHEETS)
    args = p.parse_args()

    weights = REPO / args.weights
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        sys.exit(1)

    # auto-select device
    device = args.device
    if not device:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    cache_dir = REPO / args.cache_dir
    cfg = _load_cfg()

    print("=" * 58)
    print("Phase 4 -- Steps 0-3 diagnostic run")
    print("=" * 58)
    print(f"  Weights:  {weights}")
    print(f"  Device:   {device}")
    print(f"  Sheets:   {args.sheets}")
    print(f"  Out dir:  {OUT_DIR}")
    print()

    all_stat_lines: list[str] = []

    for sheet_id in args.sheets:
        png = RAW_DIR / f"{sheet_id}.png"
        gml = RAW_DIR / f"{sheet_id}.graphml"
        if not (png.exists() and gml.exists()):
            print(f"[warn] sheet {sheet_id}: missing png or graphml, skipping")
            continue

        print(f"\n--- Sheet {sheet_id} ---")

        # Load background bboxes + GT vessel annotations (tank/pump)
        bg_bboxes, img_w, img_h, vessel_entries = _load_bg_bboxes(gml)
        print(f"  Sheet size: {img_w}x{img_h}px, "
              f"background regions: {len(bg_bboxes)}, "
              f"GT vessels (tank/pump): {len(vessel_entries)}")

        # Step 0: raw inference
        raw_preds = _load_or_infer(
            sheet_id, png, weights, args.conf, args.iou, device, cache_dir
        )
        n_raw = len(raw_preds)
        print(f"  Raw predictions: {n_raw}")

        # Step 0a: ROI filter
        preds_after_roi, n_roi_dropped = roi_filter(
            raw_preds, bg_bboxes, img_w, img_h,
            border_margin_frac=cfg["roi_border_margin_frac"],
        )
        print(f"  After ROI filter: {len(preds_after_roi)} "
              f"(-{n_roi_dropped} dropped)")

        # Step 0b: centroid NMS
        preds_deduped, n_nms_dropped = centroid_nms(
            preds_after_roi,
            centroid_frac=cfg["nms_centroid_frac"],
        )
        print(f"  After centroid-NMS: {len(preds_deduped)} "
              f"(-{n_nms_dropped} suppressed)")

        # Step 1: node set -- YOLO detections + GT vessel bodies
        nodes = build_node_set(preds_deduped)
        vessel_nodes = _build_vessel_nodes(vessel_entries, start_id=len(nodes))
        all_symbol_nodes = nodes + vessel_nodes  # combined for erasure, binding, render
        n_graph = sum(1 for n in all_symbol_nodes if n.is_graph_node)
        print(f"  Node set: {len(all_symbol_nodes)} total "
              f"({n_graph} graph nodes, "
              f"{sum(1 for n in all_symbol_nodes if n.node_type == 'flow_arrow')} arrows, "
              f"{sum(1 for n in all_symbol_nodes if n.node_type == 'tag_rect')} tag_rects, "
              f"{len(vessel_nodes)} GT vessels)")

        # Step 2: erasure -- symbols + vessels
        # Vessel bboxes use a larger dilation (vessel outlines extend to bbox edge)
        img_bgr = cv2.imread(str(png))
        if img_bgr is None:
            print(f"  ERROR: cannot read {png}")
            continue

        # Pre-erasure binary: binarize the ORIGINAL image before any symbol is erased.
        # Used by find_short_gap_pairs to confirm connectivity for close-together symbols
        # whose connecting pipe is consumed by overlapping dilated-bbox erasure zones.
        _, binary_pre_erase = _binarize_orig(img_bgr, blur_sigma=cfg["blur_sigma"],
                                              close_disk_r=0, open_disk_r=0)

        vessel_dil = cfg["erase_dilation_px"] + cfg["vessel_erase_dilation_px"]
        erased_bgr = erase_symbols(img_bgr, nodes, dilation_px=cfg["erase_dilation_px"])
        erased_bgr = erase_symbols(erased_bgr, vessel_nodes, dilation_px=vessel_dil)

        erased_out = OUT_DIR / f"sheet_{sheet_id}_erased.jpg"
        erased_out.parent.mkdir(parents=True, exist_ok=True)
        scale = cfg["render_scale"]

        # Step 3: lines
        print("  Running step 3 (binarize -> skeleton -> binding) ...")
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
        s3 = run_step3(
            erased_bgr, all_symbol_nodes, img_w, img_h,
            binary_pre_erase=binary_pre_erase,
            **step3_kwargs,
        )
        print(f"  Branches: {len(s3.branches)}, "
              f"endpoints: {len(s3.endpoints)}, "
              f"bound: {s3.n_bound}, "
              f"unbound: {s3.n_unbound}, "
              f"contests: {s3.n_contests}, "
              f"off_page: {s3.n_off_page}")

        # Step 4: junction analysis (pass binary_raw for bridge-gap detection on degree-4 nodes)
        # bg_bboxes excludes title block / legend regions from crossing analysis
        print("  Running step 4 (junction analysis) ...")
        s4 = run_step4(
            s3.skeleton, s3.branches, s3.endpoints,
            collinear_angle_tol=cfg["collinear_angle_tol"],
            min_crossing_gap=cfg["min_crossing_gap"],
            max_crossing_gap=cfg["max_crossing_gap"],
            elbow_angle_min=cfg["elbow_angle_min"],
            junction_radius_R=cfg["junction_radius_R"],
            binary_raw=s3.binary_raw,
            exclude_rects=bg_bboxes,
        )
        subtypes = s4.connector_subtypes()
        print(f"  Connectors: {s4.n_connectors} "
              f"({', '.join(f'{k}={v}' for k, v in sorted(subtypes.items()))}), "
              f"crossings: {s4.n_crossings}")

        # Step 4b: crossing-separation veto for short-gap pairs (docs/phase4_step4_scope.md §3)
        SR, Sc = build_sr_sc(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        kept_pairs, suppressed_pairs = veto_short_gap_by_crossing_separation(
            s3.short_gap_pairs, SR, Sc
        )
        if suppressed_pairs:
            print(f"  Crossing-separation veto: suppressed {len(suppressed_pairs)} "
                  f"of {len(s3.short_gap_pairs)} short-gap pairs")
        s3 = dataclasses.replace(s3, short_gap_pairs=kept_pairs)

        # Step 5: graph construction
        print("  Running step 5 (graph construction) ...")
        s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        cs = s5.components_summary()
        print(f"  Graph: {s5.n_nodes} nodes, {s5.n_edges} edges, "
              f"{s5.n_dangling} dangling branches, "
              f"{cs['n_components']} components (largest={cs['largest']})")

        # Step 6: contraction
        print("  Running step 6 (contraction) ...")
        s6 = run_step6(s5)
        cs6 = s6.components_summary()
        print(f"  Contracted: {s6.n_contracted_connectors} connectors, "
              f"{s6.n_contracted_crossings} crossings -> "
              f"{s6.n_nodes} nodes, {s6.n_edges} edges, "
              f"{cs6['n_components']} components (largest={cs6['largest']})")

        # Step 7: direction tagging from flow arrows.
        # Arrows that became transit nodes (ports > 0) are already in the contracted graph
        # as sym_N nodes; exclude them from edge-tagging to avoid self-referential direction tags.
        flow_arrows = [n for n in all_symbol_nodes
                       if n.node_type == "flow_arrow" and len(n.ports) == 0]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        print(f"  Running step 7 (direction tagging, {len(flow_arrows)} arrows) ...")
        s7 = run_step7(
            s6.graph, flow_arrows,
            img_gray=img_gray,
            max_assign_dist_px=cfg["arrow_max_dist_px"],
        )
        print(f"  Arrows: {s7.n_arrows_assigned}/{s7.n_arrows_total} assigned, "
              f"{s7.n_arrows_directed} directed, "
              f"{s7.n_edges_directed}/{s6.n_edges} edges tagged")

        # Step 8: export JSON + GraphML
        paths = run_step8(s6.graph, sheet_id, OUT_DIR)
        print(f"  Exported -> {paths['json'].name}, {paths['graphml'].name}")

        # Step 9: evaluation vs GT
        print("  Running step 9 (evaluation vs GT) ...")
        s9 = run_step9(s6.graph, gml)
        print(f"  Eval: matched {s9.n_pred_matched}/{s9.n_gt_scoreable} GT nodes "
              f"(recall {s9.match_recall:.1%})  "
              f"edge F1={s9.edge_f1:.3f}  "
              f"crossing-connect rate={s9.crossing_nonedge_rate:.1%}")

        # Save erased image (vessels already erased from erased_bgr)
        erased_for_save = erased_bgr

        erased_small = cv2.resize(
            erased_for_save,
            (int(img_w * scale), int(img_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(erased_out), erased_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Erased image -> {erased_out.name}")

        # Diagnostic render (includes junction overlays)
        diag_out = OUT_DIR / f"sheet_{sheet_id}_diag.jpg"
        render_diagnostic(
            png, all_symbol_nodes, s3.off_page_nodes, s3,
            diag_out, scale=scale, s4=s4,
        )

        # Per-sheet stats
        stat_lines = print_sheet_stats(
            sheet_id, n_raw, len(preds_after_roi), len(preds_deduped),
            all_symbol_nodes, s3.off_page_nodes, s3,
            s4=s4, s5=s5, s6=s6, s7=s7, s9=s9,
        )
        all_stat_lines.extend(stat_lines)

    # Write combined stats file
    stats_out = OUT_DIR / "stats.txt"
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text("\n".join(all_stat_lines), encoding="utf-8")
    print(f"\nStats -> {stats_out}")
    print(f"\nDone.  All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
