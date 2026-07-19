"""Phase 4 Step 4 -- decompose the P3 FP bucket into P3a/P3b/P3c sub-causes.

Measurement only. Re-derives today's production graph per sheet (0, 3, 10),
re-classifies every FP as S4/P3/OTHER exactly like scripts/rebaseline_step4.py,
then for the P3 subset (bucket == "P3", i.e. no GT path at all between the two
mapped GT nodes) determines the underlying mechanism from the edge attributes
already present in the contracted graph:

  - contracted=True, via_type in {"connector","crossing"}  -> P3c candidate
    (the edge exists ONLY because node contraction bridged two components
    that the GT graph keeps entirely separate)
  - short_gap=True (no contracted attr)                     -> P3a/b candidate,
    corridor-ink mechanism (dashed line vs neighbour-pipe bleed)
  - plain branch edge (branch_type != -1, no contracted attr) -> P3a/b candidate,
    direct skeleton-trace mechanism (continuous ink vs mis-bound branch)

For the short-gap and skeleton-branch groups we additionally print corridor /
branch geometry diagnostics (continuity ratio, nearest third-node proximity)
and save annotated crops for a sample, to support the P3a vs P3b visual call.

Usage:
    PYTHONPATH=src python scripts/decompose_p3.py --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import cv2

from pidetect.graph.build import run_step5, run_step6, sym_gid
from pidetect.graph.evaluate import load_gt_contracted, match_nodes
from pidetect.graph.lines import _corridor_ink_check, _has_aligned_corridor_path, binarize as _binarize_orig

from run_step4_veto_eval import _build_through_s4, _diagnose_risk
from run_phase4_steps03 import _load_bg_bboxes
from pidetect.graph.build import build_sr_sc

RAW_DIR = REPO / "data" / "realworld_eval" / "open100" / "_raw"
CFG_PATH = REPO / "configs" / "phase4.yaml"
OUT_DIR = REPO / "docs" / "phase4_step4_scope" / "p3_crops"
SHEETS = [0, 3, 10]


def _classify_fp(gt_u, gt_v, gt_G, violation_set) -> str:
    if frozenset((gt_u, gt_v)) in violation_set:
        return "S4"
    if not nx.has_path(gt_G, gt_u, gt_v):
        return "P3"
    return "OTHER"


def _node_by_id(all_symbol_nodes, off_page_nodes, node_id):
    for n in all_symbol_nodes + off_page_nodes:
        if n.node_id == node_id:
            return n
    return None


def _crop(png_path: Path, ax, ay, bx, by, tag: str, margin: int = 150) -> Path:
    with Image.open(png_path) as im:
        w, h = im.size
        x1, y1 = min(ax, bx), min(ay, by)
        x2, y2 = max(ax, bx), max(ay, by)
        box = (max(0, int(x1 - margin)), max(0, int(y1 - margin)),
               min(w, int(x2 + margin)), min(h, int(y2 + margin)))
        crop = im.crop(box).convert("RGB")
        draw = ImageDraw.Draw(crop)
        ox, oy = box[0], box[1]
        r = 10
        draw.ellipse([ax - ox - r, ay - oy - r, ax - ox + r, ay - oy + r], outline=(255, 0, 0), width=4)
        draw.ellipse([bx - ox - r, by - oy - r, bx - ox + r, by - oy + r], outline=(0, 120, 255), width=4)
        draw.line([ax - ox, ay - oy, bx - ox, by - oy], fill=(0, 200, 0), width=2)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{tag}.png"
        crop.save(out_path)
    return out_path


def _corridor_continuity(binary_pre_erase, a, b, all_nodes, corridor_half_width=10.0):
    """Return (bins_occ_frac, n_dark_px, third_node_frac) diagnostic for the corridor
    between a and b, computed on the TRUE production binary.

    IMPORTANT: neither scripts/run_phase4_steps03.py nor the eval-script helper
    _build_through_s4 pass mask_all_nodes/mask_all_nodes_fallback to run_step3, so
    run_step3's defaults apply (mask_all_nodes=False). The "locked C1 config"
    (mask_all_nodes: true, mask_all_nodes_fallback: false) documented in
    configs/phase4.yaml / docs/phase4_step3_final.md is NOT actually wired into either
    call site -- production runs today with ONLY A and B's own original bbox masked,
    not every node's dilated bbox. This means third-symbol-body ink and neighbour-pipe
    ink are NOT excluded from the corridor test today. This diagnostic reproduces the
    ACTUAL running behaviour (binary_pre_erase, A/B-only masking) and additionally
    flags what fraction of in-corridor dark pixels fall inside some OTHER node's
    original bbox (third_node_frac) as direct evidence of symbol-body contamination.
    """
    H, W = binary_pre_erase.shape
    rx1 = max(0, int(min(a.x1, b.x1)) - 2)
    ry1 = max(0, int(min(a.y1, b.y1)) - 2)
    rx2 = min(W, int(max(a.x2, b.x2)) + 2)
    ry2 = min(H, int(max(a.y2, b.y2)) + 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0, 0, 0.0
    crop = binary_pre_erase[ry1:ry2, rx1:rx2].copy()
    for node in (a, b):
        nx1 = max(0, int(node.x1) - rx1); ny1 = max(0, int(node.y1) - ry1)
        nx2 = min(crop.shape[1], int(node.x2) - rx1); ny2 = min(crop.shape[0], int(node.y2) - ry1)
        if nx1 < nx2 and ny1 < ny2:
            crop[ny1:ny2, nx1:nx2] = False

    third_node_mask = np.zeros_like(crop, dtype=bool)
    for node in all_nodes:
        if node.node_id in (a.node_id, b.node_id):
            continue
        if getattr(node, "node_type", "") == "flow_arrow":
            continue
        nx1 = max(0, int(node.x1) - rx1); ny1 = max(0, int(node.y1) - ry1)
        nx2 = min(crop.shape[1], int(node.x2) - rx1); ny2 = min(crop.shape[0], int(node.y2) - ry1)
        if nx1 < nx2 and ny1 < ny2:
            third_node_mask[ny1:ny2, nx1:nx2] = True

    rows, cols = np.where(crop)
    if len(rows) == 0:
        return 0.0, 0, 0.0
    px = cols.astype(float) + rx1
    py = rows.astype(float) + ry1
    dx, dy = b.cx - a.cx, b.cy - a.cy
    seg_len = math.hypot(dx, dy)
    if seg_len < 1.0:
        return 1.0, len(rows), 0.0
    ux, uy = dx / seg_len, dy / seg_len
    along = (px - a.cx) * ux + (py - a.cy) * uy
    perp_abs = np.abs((px - a.cx) * (-uy) + (py - a.cy) * ux)
    in_corr = perp_abs <= corridor_half_width
    along_c = along[in_corr]
    if len(along_c) == 0:
        return 0.0, 0, 0.0
    third_frac = float(third_node_mask[rows[in_corr], cols[in_corr]].mean())
    gap_len = float(along_c.max() - along_c.min())
    if gap_len < 2.0:
        return 1.0, int(in_corr.sum()), third_frac
    bin_size = 2.0
    n_bins = max(1, int(gap_len / bin_size))
    bins_occ = len(np.unique(np.floor((along_c - along_c.min()) / bin_size).astype(np.int32)))
    return bins_occ / n_bins, int(in_corr.sum()), third_frac


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

    all_rows = []  # (sheet, id_a, id_b, mechanism, a_class, b_class, extra)
    sr_by_sheet = {}  # sheet_id -> (SR graph, png path)

    for sheet_id in SHEETS:
        png, gml, all_symbol_nodes, s3, s4 = _build_through_s4(
            sheet_id, weights, cache_dir, args.conf, args.iou, args.device, cfg
        )
        s5 = run_step5(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        s6 = run_step6(s5)
        gt_G, crossing_infos = load_gt_contracted(gml)
        pred_to_gt, gt_to_pred = match_nodes(s6.graph, gt_G)
        gt_edge_set = {frozenset((u, v)) for u, v in gt_G.edges()}
        violation_set = {
            frozenset((a, b)) for info in crossing_infos for a, b in info["separated_pairs"]
        }

        node_by_id = {n.node_id: n for n in all_symbol_nodes + s3.off_page_nodes}

        bg_bboxes, img_w, img_h, vessel_entries = _load_bg_bboxes(gml)
        img_bgr = cv2.imread(str(png))
        _, binary_pre_erase = _binarize_orig(
            img_bgr, blur_sigma=cfg["blur_sigma"], close_disk_r=0, open_disk_r=0
        )
        SR, Sc = build_sr_sc(all_symbol_nodes, s3.off_page_nodes, s3, s4)
        sr_by_sheet[sheet_id] = (SR, png)

        for u, v, edata in s6.graph.edges(data=True):
            gt_u, gt_v = pred_to_gt.get(u), pred_to_gt.get(v)
            if gt_u is None or gt_v is None:
                continue
            if frozenset((gt_u, gt_v)) in gt_edge_set:
                continue  # TP
            bucket = _classify_fp(gt_u, gt_v, gt_G, violation_set)
            if bucket != "P3":
                continue
            id_a, id_b = int(u.split("_")[1]), int(v.split("_")[1])
            a_node = node_by_id.get(id_a)
            b_node = node_by_id.get(id_b)

            is_sg = bool(edata.get("short_gap", False))
            is_contracted = bool(edata.get("contracted", False))
            via_type = edata.get("via_type")

            if is_contracted:
                mech = f"P3c(contracted:{via_type})"
                extra = ""
            elif is_sg:
                cont_frac, n_dark, third_frac = _corridor_continuity(
                    binary_pre_erase, a_node, b_node, all_symbol_nodes
                )
                orig_gx = max(0.0, a_node.x1 - b_node.x2, b_node.x1 - a_node.x2)
                orig_gy = max(0.0, a_node.y1 - b_node.y2, b_node.y1 - a_node.y2)
                bbox_touch = (orig_gx == 0.0 and orig_gy == 0.0)
                if bbox_touch:
                    mech = "P3b(bbox_touch_no_ink)"
                    extra = "orig bboxes touch/overlap -- short-gap fired with ZERO ink evidence"
                elif third_frac >= 0.5:
                    mech = "P3b(third_node_ink)"
                    extra = (f"corridor_continuity={cont_frac:.2f} n_dark_px={n_dark} "
                              f"third_node_frac={third_frac:.2f} -- corridor ink is mostly a THIRD node's body")
                else:
                    mech = "P3ab(short_gap_ink)"
                    extra = (f"corridor_continuity={cont_frac:.2f} n_dark_px={n_dark} "
                              f"third_node_frac={third_frac:.2f}")
            else:
                length_px = edata.get("length_px", -1)
                mech = "P3ab(skeleton_branch)"
                extra = f"branch_length_px={length_px:.0f}" if length_px != -1 else ""

            a_cls = a_node.cls_name if a_node else "?"
            b_cls = b_node.cls_name if b_node else "?"
            all_rows.append((sheet_id, id_a, id_b, mech, a_cls, b_cls, extra,
                              a_node, b_node, png))

    # ------------------------------------------------------------------
    # Summary table: mechanism counts per sheet
    # ------------------------------------------------------------------
    print("=" * 100)
    print("P3 FP MECHANISM BREAKDOWN (structural, pre-visual-review)")
    print("=" * 100)
    mech_order = ["P3c(contracted:connector)", "P3c(contracted:crossing)",
                  "P3b(bbox_touch_no_ink)", "P3b(third_node_ink)",
                  "P3ab(short_gap_ink)", "P3ab(skeleton_branch)"]
    totals = {m: 0 for m in mech_order}
    for sheet_id in SHEETS:
        rows = [r for r in all_rows if r[0] == sheet_id]
        counts = {m: 0 for m in mech_order}
        for r in rows:
            counts[r[3]] = counts.get(r[3], 0) + 1
        n = len(rows)
        print(f"\nSheet {sheet_id}: total P3 FPs = {n}")
        for m in mech_order:
            c = counts.get(m, 0)
            totals[m] += c
            print(f"  {m:<30}{c:>3}  ({100*c/n:.0f}%)" if n else f"  {m:<30}{c:>3}")

    grand = sum(totals.values())
    print(f"\nTOTAL P3 FPs across sheets: {grand}")
    for m in mech_order:
        print(f"  {m:<30}{totals[m]:>3}  ({100*totals[m]/grand:.0f}%)")

    # ------------------------------------------------------------------
    # Detail dump per row (for manual P3a/P3b visual call on the ab groups)
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DETAIL -- every P3 FP with mechanism + diagnostics")
    print("=" * 100)
    for sheet_id in SHEETS:
        rows = [r for r in all_rows if r[0] == sheet_id]
        print(f"\n--- Sheet {sheet_id} ---")
        for (sid, ida, idb, mech, acls, bcls, extra, a_node, b_node, png) in sorted(rows, key=lambda r: r[3]):
            print(f"  (sym_{ida},sym_{idb}) {mech:<28} {acls:<22} {bcls:<22} {extra}")

    # ------------------------------------------------------------------
    # Crops: contracted-bridge (P3c) examples, and short_gap/skeleton (P3a/b) examples
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SAVING CROPS")
    print("=" * 100)

    c_rows = [r for r in all_rows if r[3].startswith("P3c")]
    bbox_rows = [r for r in all_rows if r[3] == "P3b(bbox_touch_no_ink)"]
    third_rows = [r for r in all_rows if r[3] == "P3b(third_node_ink)"]
    ab_sg_rows = [r for r in all_rows if r[3] == "P3ab(short_gap_ink)"]
    ab_sk_rows = [r for r in all_rows if r[3] == "P3ab(skeleton_branch)"]

    def save_n(rows, n, label):
        for r in rows[:n]:
            (sid, ida, idb, mech, acls, bcls, extra, a_node, b_node, png) = r
            tag = f"sheet{sid}_{label}_sym{ida}_sym{idb}"
            path = _crop(png, a_node.cx, a_node.cy, b_node.cx, b_node.cy, tag)
            print(f"  {label}: sheet {sid} (sym_{ida},sym_{idb}) [{acls}/{bcls}] {extra} -> {path}")

    save_n(c_rows, 6, "P3c")
    save_n(bbox_rows, 3, "P3b_bboxtouch")
    save_n(third_rows, 4, "P3b_thirdnode")
    save_n(ab_sg_rows, len(ab_sg_rows), "P3ab_shortgap_ink")
    save_n(ab_sk_rows, 6, "P3ab_skelbranch")

    # ------------------------------------------------------------------
    # P3c junction investigation: SR shortest path + junction crop for every
    # contraction-bridge pair (which node(s) did contraction bridge through?)
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("P3c JUNCTION INVESTIGATION -- SR path + junction identity for every contracted-bridge FP")
    print("=" * 100)
    for r in c_rows:
        (sid, ida, idb, mech, acls, bcls, extra, a_node, b_node, png) = r
        SR, sr_png = sr_by_sheet[sid]
        gid_a, gid_b = sym_gid(ida), sym_gid(idb)
        path, culprits = _diagnose_risk(SR, gid_a, gid_b)
        junc_nodes = [n for n in path if n.startswith("junc_")]
        print(f"\nSheet {sid}: sym_{ida} ({acls}) <-> sym_{idb} ({bcls})  [{mech}]")
        print(f"  SR path: {' -> '.join(path)}")
        for jn in junc_nodes:
            d = SR.nodes[jn]
            print(f"    {jn}: node_type={d.get('node_type')} subtype={d.get('subtype')} "
                  f"cx={d.get('cx'):.0f} cy={d.get('cy'):.0f}")
            crop_path = _crop(sr_png, d["cx"], d["cy"], d["cx"], d["cy"],
                               f"sheet{sid}_P3c_junction_{jn}", margin=200)
            print(f"    crop -> {crop_path}")


if __name__ == "__main__":
    main()
