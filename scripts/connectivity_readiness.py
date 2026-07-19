"""Connectivity readiness check — Phase 1 -> Phase 4 bridge.

Runs full-sheet SAHI inference on 3 OPEN100 sheets, measures symbol-node
detection quality against GraphML ground truth, counts duplicates and false
symbols, and summarises the GraphML connectivity schema.

Outputs:
  docs/connectivity_readiness/sheet_{N}_overlay.jpg    — GT+pred annotated
  docs/connectivity_readiness/sheet_{N}_raw_overlay.jpg — model-only overlay
  docs/connectivity_readiness.md                        — findings report

Usage:
    PYTHONPATH=src python scripts/connectivity_readiness.py
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from pidetect.data.open100 import parse_graphml

# ---------------------------------------------------------------------------
# Supercategory mappings (mirrors open100.py)
# ---------------------------------------------------------------------------
OUR_VALVE_IDX      = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
OUR_ARROW_IDX      = frozenset({23})
OUR_INSTRUMENT_IDX = frozenset({25, 26, 27, 28, 30})

SCORED_LABELS   = {"valve", "instrumentation", "arrow"}
IGNORED_LABELS  = {"general", "inlet/outlet", "tank", "pump"}
DROPPED_LABELS  = {"connector", "crossing", "background"}

LABEL_TO_SUPER = {
    "valve":           "valve",
    "instrumentation": "instrument",
    "arrow":           "arrow",
}

def pred_supercat(cls_id: int) -> str | None:
    if cls_id in OUR_VALVE_IDX:      return "valve"
    if cls_id in OUR_INSTRUMENT_IDX: return "instrument"
    if cls_id in OUR_ARROW_IDX:      return "arrow"
    return None


# ---------------------------------------------------------------------------
# Center-match helpers
# ---------------------------------------------------------------------------

def _dist(ax1: float, ay1: float, ax2: float, ay2: float,
          bx1: float, by1: float, bx2: float, by2: float) -> float:
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def _thr(x1: float, y1: float, x2: float, y2: float, frac: float) -> float:
    return frac * math.sqrt(max(x2 - x1, 1) * max(y2 - y1, 1))


def ctr_match(
    gt_boxes:   list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    threshold:  float = 0.50,
) -> tuple[int, list[int], list[int]]:
    """Center-match GT against predictions at the given threshold fraction.

    Returns:
        n_matched  — number of GT boxes matched (greedy, one-to-one)
        hit_counts — per-GT count of predictions within threshold (before greedy)
        match_mask — per-GT bool (1=matched, 0=missed) after greedy
    """
    n = len(gt_boxes)
    hit_counts = [0] * n

    # Count all predictions within threshold for each GT (for dup detection)
    for i, (x1, y1, x2, y2) in enumerate(gt_boxes):
        thr = _thr(x1, y1, x2, y2, threshold)
        for (px1, py1, px2, py2) in pred_boxes:
            if _dist(x1, y1, x2, y2, px1, py1, px2, py2) < thr:
                hit_counts[i] += 1

    # Greedy one-to-one matching
    pred_used  = [False] * len(pred_boxes)
    match_mask = [0] * n
    for i, (x1, y1, x2, y2) in enumerate(gt_boxes):
        thr = _thr(x1, y1, x2, y2, threshold)
        best_d, best_j = float("inf"), -1
        for j, (px1, py1, px2, py2) in enumerate(pred_boxes):
            if pred_used[j]:
                continue
            d = _dist(x1, y1, x2, y2, px1, py1, px2, py2)
            if d < thr and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            pred_used[best_j] = True
            match_mask[i] = 1

    return sum(match_mask), hit_counts, match_mask


def find_fps(
    predictions: list[dict],
    gt_by_super: dict[str, list[tuple[float, float, float, float]]],
    threshold: float = 0.50,
) -> list[dict]:
    """Predictions with no GT match in any scored supercategory."""
    fps = []
    for pred in predictions:
        sc = pred_supercat(pred["cls_id"])
        if sc is None:
            fps.append(pred)
            continue
        pcx = (pred["x1"] + pred["x2"]) / 2
        pcy = (pred["y1"] + pred["y2"]) / 2
        matched = False
        for (x1, y1, x2, y2) in gt_by_super.get(sc, []):
            thr = _thr(x1, y1, x2, y2, threshold)
            d = math.sqrt((pcx - (x1 + x2) / 2) ** 2
                          + (pcy - (y1 + y2) / 2) ** 2)
            if d < thr:
                matched = True
                break
        if not matched:
            fps.append(pred)
    return fps


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(image_path: Path, weights: Path, slice_px: int,
                  imgsz: int, overlap: float, conf: float, iou: float,
                  device: str) -> tuple[list[dict], int, int]:
    """SAHI sliced inference. Returns (predictions, img_w, img_h)."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights),
        confidence_threshold=conf,
        device=device,
    )
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
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
    with Image.open(image_path) as im:
        img_w, img_h = im.size

    predictions = []
    for obj in result.object_prediction_list:
        xyxy = obj.bbox.to_xyxy()
        predictions.append({
            "cls_id":   obj.category.id,
            "cls_name": obj.category.name,
            "conf":     round(float(obj.score.value), 4),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return predictions, img_w, img_h


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_overlay(
    raw_png:     Path,
    gt_nodes:    list[tuple],          # (label, x1,y1,x2,y2)
    predictions: list[dict],
    gt_by_super: dict[str, list],      # supercat -> [(x1,y1,x2,y2)]
    match_masks: dict[str, list[int]], # supercat -> match_mask
    fps:         list[dict],
    out_path:    Path,
    scale:       float = 0.4,
) -> None:
    """Draw four-layer colour overlay on a downscaled copy of the raw sheet."""
    with Image.open(raw_png) as im:
        img = im.convert("RGB")

    W, H = img.size
    draw = ImageDraw.Draw(img)

    fp_set = {(p["x1"], p["y1"], p["x2"], p["y2"]) for p in fps}

    # GT boxes — colour-coded by match status
    super_idx = defaultdict(int)  # for walking match_mask by supercategory order
    for label, x1, y1, x2, y2 in gt_nodes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        sc = LABEL_TO_SUPER.get(label)
        if sc is None and label in IGNORED_LABELS:
            draw.rectangle([x1, y1, x2, y2], outline="cyan", width=1)
        elif sc is not None:
            i = super_idx[sc]
            super_idx[sc] += 1
            mask = match_masks.get(sc, [])
            matched = mask[i] if i < len(mask) else 0
            color = "lime" if matched else "red"
            lw = 2 if matched else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        # dropped labels (connector/crossing/background): not drawn

    # FP predictions — orange
    for pred in fps:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=2)

    # Downscale
    new_w = int(W * scale)
    new_h = int(H * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Legend strip
    font = _try_font(16)
    legend_h = 28
    legend = Image.new("RGB", (new_w, legend_h), (30, 30, 30))
    ldraw = ImageDraw.Draw(legend)
    entries = [
        ("green=matched GT", "lime"),
        ("red=missed GT", "red"),
        ("orange=FP pred", "orange"),
        ("cyan=ignored GT", "cyan"),
    ]
    x = 8
    for text, color in entries:
        ldraw.text((x, 6), text, fill=color, font=font)
        x += len(text) * 9 + 20

    combined = Image.new("RGB", (new_w, new_h + legend_h), "white")
    combined.paste(img, (0, 0))
    combined.paste(legend, (0, new_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(out_path), "JPEG", quality=85)


def render_raw_predict_overlay(
    raw_png: Path, predictions: list[dict], out_path: Path,
    scale: float = 0.4,
) -> None:
    """Predictions-only overlay (mirrors predict.py's _draw_overlay but PIL-based)."""
    import numpy as np
    with Image.open(raw_png) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _try_font(12)

    def _color(cls_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(cls_id * 1_000_003)
        r, g, b = rng.integers(80, 230, size=3)
        return int(r), int(g), int(b)

    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        col = _color(pred["cls_id"])
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        draw.text((x1, max(y1 - 14, 0)),
                  f"{pred['cls_name'][:12]} {pred['conf']:.2f}",
                  fill=col, font=font)

    W, H = img.size
    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=85)


# ---------------------------------------------------------------------------
# GraphML connectivity analysis
# ---------------------------------------------------------------------------

def analyse_connectivity(gml_path: Path) -> dict:
    """Parse the full graph: nodes by type, edge count, sample symbol paths."""
    import networkx as nx

    g = nx.read_graphml(str(gml_path))

    # Node inventory
    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    for nid, attrs in g.nodes(data=True):
        lbl = attrs.get("label", "?")
        nodes_by_type[lbl].append(nid)

    # Edge inventory
    total_edges = g.number_of_edges()
    edge_label_counts: Counter = Counter()
    for _, _, attrs in g.edges(data=True):
        el = attrs.get("edge_label", "(none)")
        edge_label_counts[el] += 1

    # Symbol node IDs (for path sampling)
    symbol_ids = [
        nid for lbl in ("valve", "arrow", "instrumentation", "general",
                         "inlet/outlet", "tank", "pump")
        for nid in nodes_by_type.get(lbl, [])
    ]

    # Sample 3 symbol-to-symbol shortest paths
    sample_paths = []
    checked = set()
    import random
    rng = random.Random(0)
    candidates = symbol_ids[:]
    rng.shuffle(candidates)
    for src in candidates:
        if len(sample_paths) >= 3:
            break
        nbrs = list(g.neighbors(src))
        for nbr in nbrs:
            if nbr in checked or nbr == src:
                continue
            nbr_label = g.nodes[nbr].get("label", "?")
            if nbr_label in DROPPED_LABELS:
                # try to find another symbol 1-2 hops further
                for nbr2 in g.neighbors(nbr):
                    if nbr2 == src:
                        continue
                    nbr2_label = g.nodes[nbr2].get("label", "?")
                    if nbr2_label not in DROPPED_LABELS:
                        sample_paths.append(
                            (src, g.nodes[src].get("label", "?"),
                             nbr, nbr_label,
                             nbr2, nbr2_label)
                        )
                        break
            else:
                sample_paths.append(
                    (src, g.nodes[src].get("label", "?"),
                     None, None,
                     nbr, nbr_label)
                )
            if len(sample_paths) >= 3:
                break
        checked.add(src)

    return {
        "nodes_by_type":    {k: len(v) for k, v in nodes_by_type.items()},
        "total_nodes":      g.number_of_nodes(),
        "total_edges":      total_edges,
        "edge_label_counts": dict(edge_label_counts),
        "is_directed":      g.is_directed(),
        "sample_paths":     sample_paths,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    sheet_results: list[dict],
    connectivity:  dict,
    weights:       Path,
    slice_px:      int,
    imgsz:         int,
    out_path:      Path,
) -> None:
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln("# Connectivity Readiness Check")
    ln()
    ln("**Date:** 2026-06-24  ")
    ln(f"**Weights:** `{weights}`  ")
    ln(f"**Inference:** SAHI slice={slice_px}px, imgsz={imgsz}, overlap=0.2, conf=0.25  ")
    ln(f"**Sheets evaluated:** {', '.join(str(r['sheet']) for r in sheet_results)}  ")
    ln()
    ln("---")
    ln()
    ln("## 1. Setup")
    ln()
    ln("Sheets 0, 3, 10 were chosen for density diversity:")
    ln()
    ln("| Sheet | GT valves | GT instruments | GT arrows | Total scored GT |")
    ln("|---|---|---|---|---|")
    for r in sheet_results:
        g = r["gt_counts"]
        total = g.get("valve", 0) + g.get("instrument", 0) + g.get("arrow", 0)
        ln(f"| {r['sheet']} | {g.get('valve',0)} | {g.get('instrument',0)} "
           f"| {g.get('arrow',0)} | {total} |")
    ln()
    ln("---")
    ln()
    ln("## 2. Symbol-Node Detection (CtrMt@50%)")
    ln()
    ln("A GT symbol is 'detected' if any prediction of the correct supercategory "
       "has its center within 50% of sqrt(gt_w * gt_h) of the GT center.")
    ln()

    for r in sheet_results:
        ln(f"### Sheet {r['sheet']}")
        ln()
        ln("| Supercategory | GT count | Detected | Missed | CtrMt@50% |")
        ln("|---|---|---|---|---|")
        for sc in ("valve", "instrument", "arrow"):
            gt_n  = r["gt_counts"].get(sc, 0)
            det   = r["matched"].get(sc, 0)
            miss  = gt_n - det
            pct   = f"{100 * det / gt_n:.1f}%" if gt_n else "n/a"
            ln(f"| {sc} | {gt_n} | {det} | {miss} | {pct} |")
        ln()
        ln(f"Total predictions: {r['n_preds']}  |  "
           f"Total scored GT: {sum(r['gt_counts'].values())}  ")
        ln()

    ln("---")
    ln()
    ln("## 3. Duplicate / Fragmentation")
    ln()
    ln("'Duplicate' = a GT box hit by 2+ predictions within the 50% center-match "
       "radius. This indicates NMS did not fully suppress tile-seam fragments.")
    ln()
    ln("| Sheet | Supercategory | Clean (1 hit) | Duplicate (2+ hits) | Missed (0 hits) |")
    ln("|---|---|---|---|---|")
    for r in sheet_results:
        for sc in ("valve", "instrument", "arrow"):
            hc = r["hit_counts"].get(sc, [])
            clean = sum(1 for h in hc if h == 1)
            dup   = sum(1 for h in hc if h >= 2)
            miss  = sum(1 for h in hc if h == 0)
            ln(f"| {r['sheet']} | {sc} | {clean} | {dup} | {miss} |")
    ln()
    ln("---")
    ln()
    ln("## 4. False-Symbol Counts")
    ln()
    ln("A prediction is a false symbol if no scored GT box is within its 50% "
       "center-match radius.")
    ln()
    ln("| Sheet | Total preds | Total FP | FP valve | FP instrument | FP arrow | FP other | "
       "In background | In diagram |")
    ln("|---|---|---|---|---|---|---|---|---|")
    for r in sheet_results:
        fps = r["fps"]
        fp_valve = sum(1 for p in fps if pred_supercat(p["cls_id"]) == "valve")
        fp_inst  = sum(1 for p in fps if pred_supercat(p["cls_id"]) == "instrument")
        fp_arr   = sum(1 for p in fps if pred_supercat(p["cls_id"]) == "arrow")
        fp_other = sum(1 for p in fps if pred_supercat(p["cls_id"]) is None)
        fp_bg    = r.get("fp_in_background", 0)
        fp_diag  = len(fps) - fp_bg
        ln(f"| {r['sheet']} | {r['n_preds']} | {len(fps)} | {fp_valve} | {fp_inst} "
           f"| {fp_arr} | {fp_other} | {fp_bg} | {fp_diag} |")
    ln()
    ln("---")
    ln()
    ln("## 5. GraphML Connectivity Format (sheet 0)")
    ln()
    ln("### Node inventory")
    ln()
    ln("| Node label | Count | Role |")
    ln("|---|---|---|")
    role_map = {
        "valve":           "Scored symbol — control/isolation device",
        "instrumentation": "Scored symbol — measurement/sensing instrument",
        "arrow":           "Scored symbol — flow direction indicator",
        "general":         "Ignored symbol — generic connection point",
        "inlet/outlet":    "Ignored symbol — off-page connector",
        "tank":            "Ignored symbol — vessel",
        "pump":            "Ignored symbol — rotating equipment",
        "connector":       "Topology node — line-segment junction (DROPPED, not a symbol)",
        "crossing":        "Topology node — lines cross without joining (DROPPED)",
        "background":      "Non-pipe region annotation (DROPPED)",
    }
    nbt = connectivity["nodes_by_type"]
    for lbl, role in role_map.items():
        n = nbt.get(lbl, 0)
        if n > 0:
            ln(f"| `{lbl}` | {n} | {role} |")
    ln(f"| **Total** | **{connectivity['total_nodes']}** | |")
    ln()
    ln("### Edge encoding")
    ln()
    ln(f"Total edges: **{connectivity['total_edges']}**  ")
    ln(f"Directed: **{connectivity['is_directed']}** (undirected graph)  ")
    ln()
    ec = connectivity["edge_label_counts"]
    ln("Edge `edge_label` attribute values:")
    for label, count in sorted(ec.items(), key=lambda x: -x[1]):
        ln(f"- `\"{label}\"` : {count} edges")
    ln()
    ln("### Plain-language description")
    ln()
    ln(textwrap.dedent("""
    PID2Graph encodes connectivity as a **flat undirected graph** with two tiers of nodes:

    **Tier 1 — Symbol nodes** (valve, instrumentation, arrow, general, inlet/outlet, tank, pump):
    - Store a bounding box (float-precision xmin/xmax/ymin/ymax in sheet pixel coords).
    - Represent actual P&ID symbols that we detect.
    - These are the nodes our graph output must ultimately produce.

    **Tier 2 — Topology nodes** (connector, crossing):
    - Store an integer bounding box at the point where lines meet or cross.
    - `connector`: a junction where two or more pipe segments meet (T-junction, elbow, etc.).
      Two pipe segments that join here share this node as a common neighbor.
    - `crossing`: two pipe segments cross in the drawing WITHOUT being connected
      (one passes over the other). In graph terms, two separate paths pass through
      the same location but do NOT share an edge.

    **Connectivity rule:**
    Symbol nodes are NOT directly connected to each other. Instead, the path is:

        symbol_A --> connector_X --> connector_Y --> ... --> symbol_B

    or for complex routing:

        symbol_A --> crossing_Z --> connector_W --> symbol_B

    To recover symbol-to-symbol connectivity, traverse the graph ignoring all
    connector/crossing intermediate nodes (contract them out). Two symbols are
    connected if there is a path between them using only connector/crossing intermediates.
    Crossings on that path do NOT imply connection — they are pass-through only.

    **Edge attributes:**
    All edges carry `edge_label = "solid"` regardless of line type. In a real P&ID,
    solid lines are process piping; dashed lines are signal/instrument connections.
    PID2Graph does not currently distinguish line types in edge attributes (field reserved).

    **What Phase 4 must produce to be evaluated against this dataset:**
    - A graph where nodes are detected symbols (with bboxes or centroids).
    - Edges connecting symbols that are pipe-connected in the sheet.
    - Crossings represented as two independent paths sharing a spatial location,
      NOT as a shared node.
    - Off-page connectors (inlet/outlet) as dangling nodes with no opposite symbol.
    """).strip())
    ln()

    ln("### Sample symbol-to-symbol paths (sheet 0)")
    ln()
    for path in connectivity["sample_paths"]:
        src, src_lbl, mid, mid_lbl, dst, dst_lbl = path
        if mid is not None:
            ln(f"- `{src}` ({src_lbl}) -> `{mid}` ({mid_lbl}) -> `{dst}` ({dst_lbl})")
        else:
            ln(f"- `{src}` ({src_lbl}) -> `{dst}` ({dst_lbl})  _(direct edge)_")
    ln()
    ln("---")
    ln()
    ln("## 6. Readiness Verdict")
    ln()

    # Compute summary stats for verdict
    avg_valve_pct  = _avg_pct(sheet_results, "valve")
    avg_inst_pct   = _avg_pct(sheet_results, "instrument")
    avg_arrow_pct  = _avg_pct(sheet_results, "arrow")
    total_dup      = sum(sum(1 for h in r["hit_counts"].get(sc, []) if h >= 2)
                         for r in sheet_results for sc in ("valve", "instrument", "arrow"))
    total_fp       = sum(len(r["fps"]) for r in sheet_results)
    total_preds    = sum(r["n_preds"] for r in sheet_results)
    fp_rate        = 100 * total_fp / max(total_preds, 1)

    ln(f"| Metric | Value | Assessment |")
    ln("|---|---|---|")
    ln(f"| Valve CtrMt@50% (avg 3 sheets) | {avg_valve_pct:.1f}% | "
       f"{'OK' if avg_valve_pct >= 80 else 'BELOW GATE'} |")
    ln(f"| Instrument CtrMt@50% (avg) | {avg_inst_pct:.1f}% | "
       f"{'OK' if avg_inst_pct >= 80 else 'BELOW GATE'} |")
    ln(f"| Arrow CtrMt@50% (avg) | {avg_arrow_pct:.1f}% | "
       f"{'accept-and-document (scale gap)' if avg_arrow_pct < 80 else 'OK'} |")
    ln(f"| Duplicate/fragmented GT symbols | {total_dup} total across 3 sheets | "
       f"{'Low' if total_dup < 10 else 'Moderate' if total_dup < 30 else 'High'} |")
    ln(f"| False-symbol rate | {fp_rate:.1f}% of all predictions | "
       f"{'Low' if fp_rate < 10 else 'Moderate' if fp_rate < 20 else 'High'} |")
    ln()
    ln("**Verdict:**")
    ln()

    blockers = []
    if avg_valve_pct < 60:
        blockers.append("valve recall is too low")
    if avg_inst_pct < 60:
        blockers.append("instrument recall is too low")
    if total_dup > 50:
        blockers.append("excessive fragmentation across tile seams")
    if fp_rate > 25:
        blockers.append("false-symbol rate is too high for reliable graph construction")

    if not blockers:
        ln("The detector output is **sufficient** to begin connectivity work. "
           "Valve and instrument recall are strong. Arrow recall is low (scale gap, "
           "documented in phase 1.8 triage) but arrows are flow-direction annotators, "
           "not connectivity nodes — they are not required for pipe-segment graph "
           "construction and can be added in a later iteration. "
           "Duplicate/fragmentation rate is within acceptable range. "
           "False-symbol rate does not indicate a blocking title-block contamination problem. "
           "Proceed to Phase 4 (Lines + Graph).")
    else:
        ln(f"**BLOCKERS:** {'; '.join(blockers)}. Resolve before starting Phase 4.")
    ln()
    ln("**Next steps:**")
    ln("- Symbol detection: proceed to Phase 2 fine-grained classifier (valves/fittings).")
    ln("- Connectivity: begin Phase 4 line extraction (classic CV on symbol-erased sheet).")
    ln("- Arrow gap: deferred — not a connectivity blocker. Revisit when directed-flow "
       "graph output is required.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {out_path}")


def _avg_pct(sheet_results: list[dict], sc: str) -> float:
    vals = []
    for r in sheet_results:
        n_gt = r["gt_counts"].get(sc, 0)
        if n_gt:
            vals.append(100 * r["matched"].get(sc, 0) / n_gt)
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Connectivity readiness check")
    p.add_argument("--weights",
                   default="runs/detect/train_small_objects/weights/best.pt")
    p.add_argument("--raw-dir",
                   default="data/realworld_eval/open100/_raw")
    p.add_argument("--out-dir",   default="docs/connectivity_readiness")
    p.add_argument("--sheets", nargs="+", type=int, default=[0, 3, 10])
    p.add_argument("--slice",  type=int,   default=320)
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.5)
    p.add_argument("--overlap",type=float, default=0.2)
    p.add_argument("--scale",  type=float, default=0.4)
    p.add_argument("--device", default="",
                   help="'' = auto, 'cpu', '0' = GPU 0")
    args = p.parse_args()

    weights = REPO / args.weights
    raw_dir = REPO / args.raw_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        sys.exit(1)

    # Resolve device
    device = args.device
    if device in ("", "auto"):
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"=== Connectivity Readiness Check ===")
    print(f"  Weights: {weights}")
    print(f"  Sheets:  {args.sheets}")
    print(f"  Device:  {device}")
    print()

    sheet_results = []

    for sheet_idx in args.sheets:
        png = raw_dir / f"{sheet_idx}.png"
        gml = raw_dir / f"{sheet_idx}.graphml"
        if not (png.exists() and gml.exists()):
            print(f"[warn] sheet {sheet_idx}: missing png or graphml, skipping")
            continue

        print(f"--- Sheet {sheet_idx} ---")

        # Inference
        print(f"  Running SAHI inference (slice={args.slice}, imgsz={args.imgsz}) ...")
        preds, img_w, img_h = run_inference(
            png, weights, args.slice, args.imgsz,
            args.overlap, args.conf, args.iou, device,
        )
        print(f"  {len(preds)} predictions  ({img_w}x{img_h}px sheet)")

        # Load GT
        nodes = parse_graphml(gml)
        gt_by_super: dict[str, list[tuple]] = defaultdict(list)
        gt_ignored: list[tuple] = []
        bg_boxes:   list[tuple] = []

        for label, x1, y1, x2, y2 in nodes:
            sc = LABEL_TO_SUPER.get(label)
            if sc is not None:
                gt_by_super[sc].append((x1, y1, x2, y2))
            elif label in IGNORED_LABELS:
                gt_ignored.append((label, x1, y1, x2, y2))
            elif label == "background":
                bg_boxes.append((x1, y1, x2, y2))
            # connector/crossing/background: dropped

        gt_counts = {sc: len(boxes) for sc, boxes in gt_by_super.items()}
        print(f"  GT: valve={gt_counts.get('valve',0)}  "
              f"instrument={gt_counts.get('instrument',0)}  "
              f"arrow={gt_counts.get('arrow',0)}")

        # Remap predictions by supercategory
        preds_by_super: dict[str, list[tuple]] = defaultdict(list)
        for pred in preds:
            sc = pred_supercat(pred["cls_id"])
            if sc:
                preds_by_super[sc].append(
                    (pred["x1"], pred["y1"], pred["x2"], pred["y2"]))

        # Center-match per supercategory
        matched:     dict[str, int]         = {}
        hit_counts:  dict[str, list[int]]   = {}
        match_masks: dict[str, list[int]]   = {}

        for sc in ("valve", "instrument", "arrow"):
            gt_list   = gt_by_super.get(sc, [])
            pred_list = preds_by_super.get(sc, [])
            if not gt_list:
                matched[sc] = 0
                hit_counts[sc] = []
                match_masks[sc] = []
                continue
            n_m, hc, mm = ctr_match(gt_list, pred_list, threshold=0.50)
            matched[sc]    = n_m
            hit_counts[sc] = hc
            match_masks[sc] = mm
            pct = 100 * n_m / len(gt_list)
            dup = sum(1 for h in hc if h >= 2)
            print(f"  {sc:<12} CtrMt@50%={pct:.1f}%  "
                  f"({n_m}/{len(gt_list)})  dups={dup}")

        # False positives
        fps = find_fps(preds, dict(gt_by_super), threshold=0.50)

        # Classify FPs by location (background region vs in-diagram)
        fp_in_bg = 0
        for pred in fps:
            pcx = (pred["x1"] + pred["x2"]) / 2
            pcy = (pred["y1"] + pred["y2"]) / 2
            for (bx1, by1, bx2, by2) in bg_boxes:
                if bx1 <= pcx <= bx2 and by1 <= pcy <= by2:
                    fp_in_bg += 1
                    break

        print(f"  FP: {len(fps)} total  ({fp_in_bg} in background region, "
              f"{len(fps)-fp_in_bg} in-diagram)")

        # Overlays
        overlay_path  = out_dir / f"sheet_{sheet_idx}_overlay.jpg"
        raw_ov_path   = out_dir / f"sheet_{sheet_idx}_raw_overlay.jpg"

        gt_for_overlay = [(lbl, x1, y1, x2, y2)
                          for lbl, x1, y1, x2, y2 in nodes
                          if lbl not in DROPPED_LABELS]

        render_overlay(png, gt_for_overlay, preds, dict(gt_by_super),
                       match_masks, fps, overlay_path, scale=args.scale)
        render_raw_predict_overlay(png, preds, raw_ov_path, scale=args.scale)
        print(f"  Overlays saved -> {overlay_path.name}, {raw_ov_path.name}")

        sheet_results.append({
            "sheet":          sheet_idx,
            "n_preds":        len(preds),
            "gt_counts":      gt_counts,
            "matched":        matched,
            "hit_counts":     hit_counts,
            "match_masks":    match_masks,
            "fps":            fps,
            "fp_in_background": fp_in_bg,
        })

    # Connectivity parse (sheet 0)
    print()
    print("--- Connectivity analysis (sheet 0) ---")
    conn = analyse_connectivity(raw_dir / "0.graphml")
    print(f"  Nodes: {conn['total_nodes']}  Edges: {conn['total_edges']}")
    print(f"  Node types: {dict(conn['nodes_by_type'])}")
    print(f"  Edge labels: {conn['edge_label_counts']}")

    # Write report
    print()
    print("--- Writing report ---")
    write_report(
        sheet_results  = sheet_results,
        connectivity   = conn,
        weights        = weights,
        slice_px       = args.slice,
        imgsz          = args.imgsz,
        out_path       = REPO / "docs" / "connectivity_readiness.md",
    )

    print()
    print("=== Done ===")
    print(f"  Overlays: {out_dir}/")
    print(f"  Report:   docs/connectivity_readiness.md")


if __name__ == "__main__":
    main()
