"""Phase 5 — end-to-end pipeline for one uploaded P&ID sheet.

Wraps the existing Phase 4 (detection -> connectivity graph) and Phase 3 (tag OCR)
public APIs for API use (see pidetect.api). No new detection/graph/OCR logic here --
this only orchestrates existing functions, mirroring scripts/run_phase4_steps03.py's
steps 0-8, adapted for a sheet with no GraphML ground truth (a real upload, not an
OPEN100 eval sheet):

  - roi_filter() is called with an empty background-bbox list, so it falls back to
    the border-margin heuristic (no GT background regions to exclude).
  - No GT vessel nodes are injected -- tanks/pumps are not a YOLO-detected class in
    this pipeline (a known v1 limitation; see README "Known limitations").
  - Step 9 (eval vs GT) does not apply to an upload with no ground truth and is not run.

Public API:
    run_pipeline(image_path, weights_path, phase4_cfg, phase3_cfg, device, stage_cb)
        -> networkx.Graph  (contracted, symbol-to-symbol, tag_* attrs on instrument nodes)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Optional

import cv2
import networkx as nx

from pidetect.detect.predict import run_sliced_inference
from pidetect.graph.erase import (
    assert_no_duplicate_scored_nodes,
    build_node_set,
    centroid_nms,
    erase_symbols,
    roi_filter,
    scored_family_group_key,
)
from pidetect.graph.lines import assert_run_step3_wired, binarize, run_step3
from pidetect.graph.junction import run_step4
from pidetect.graph.build import (
    build_sr_sc,
    run_step5,
    run_step6,
    run_step7,
    veto_short_gap_by_crossing_separation,
)

StageCb = Callable[[str, Optional[dict]], None]


def run_pipeline(
    image_path: Path,
    weights_path: Path,
    phase4_cfg: dict,
    phase3_cfg: dict,
    device: str = "cpu",
    stage_cb: Optional[StageCb] = None,
) -> nx.Graph:
    """Run detection -> dedupe -> node set -> tag OCR -> connectivity graph on one sheet.

    `stage_cb(stage_name, progress)` is called at each stage transition; `progress` is
    `None` except during "reading_tags", where it's `{"current": i, "total": n}` per
    bubble (OCR dominates per-sheet latency -- docs/phase5_design.md -- so it's the one
    stage worth surfacing fine-grained progress for).

    Returns the contracted (step 6) graph: nodes carry node_type/cls_name/bbox/conf,
    instrument nodes additionally carry tag_function/tag_loop_number/tag_raw_text/
    tag_confidence/tag_parse_status.
    """
    def _stage(name: str, progress: Optional[dict] = None) -> None:
        if stage_cb is not None:
            stage_cb(name, progress)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_h, img_w = img_bgr.shape[:2]

    _stage("detecting")
    raw_preds = run_sliced_inference(
        image_path, weights_path,
        conf=phase4_cfg.get("detect_conf", 0.25),
        iou=phase4_cfg.get("detect_iou", 0.5),
        device=device,
    )

    preds_after_roi, _ = roi_filter(
        raw_preds, [], img_w, img_h,
        border_margin_frac=phase4_cfg["roi_border_margin_frac"],
    )
    preds_deduped, _ = centroid_nms(
        preds_after_roi,
        centroid_frac=phase4_cfg["nms_centroid_frac"],
        group_key=scored_family_group_key,
    )
    nodes = build_node_set(preds_deduped)
    # Construction-time regression guard (docs/phase4_final.md §2): a cross-subtype
    # duplicate bubble/valve must never reach OCR or the graph under two identities.
    assert_no_duplicate_scored_nodes(nodes, centroid_frac=phase4_cfg["nms_centroid_frac"])

    n_instruments = sum(1 for n in nodes if n.node_type == "instrument")
    _stage("reading_tags", {"current": 0, "total": n_instruments})
    from pidetect.text.ocr import run_ocr_on_nodes  # local import: paddleocr optional at import time

    def _ocr_progress(i: int, total: int) -> None:
        _stage("reading_tags", {"current": i, "total": total})

    run_ocr_on_nodes(img_bgr, nodes, phase3_cfg, progress_cb=_ocr_progress)

    _stage("tracing_lines")
    _, binary_pre_erase = binarize(
        img_bgr, blur_sigma=phase4_cfg["blur_sigma"], close_disk_r=0, open_disk_r=0,
    )
    erased_bgr = erase_symbols(img_bgr, nodes, dilation_px=phase4_cfg["erase_dilation_px"])

    step3_kwargs = dict(
        blur_sigma=phase4_cfg["blur_sigma"],
        close_disk_r=phase4_cfg["morph_close_disk"],
        open_disk_r=phase4_cfg["morph_open_disk"],
        min_branch_len_px=phase4_cfg["min_branch_len_px"],
        endpoint_bind_extra_px=phase4_cfg["endpoint_bind_extra_px"],
        off_page_border_px=phase4_cfg["off_page_border_px"],
        mask_all_nodes=phase4_cfg["mask_all_nodes"],
        mask_all_nodes_fallback=phase4_cfg["mask_all_nodes_fallback"],
        short_gap_max_px=phase4_cfg["short_gap_max_px"],
        interpose_tol_px=phase4_cfg["short_gap_interpose_tol_px"],
        corridor_half_width_px=phase4_cfg["short_gap_corridor_half_width_px"],
        continuity_frac=phase4_cfg["short_gap_continuity_frac"],
        perp_reject_ratio=phase4_cfg["short_gap_perp_reject_ratio"],
        stub_angle_tol_deg=phase4_cfg["short_gap_stub_angle_tol_deg"],
    )
    assert_run_step3_wired(phase4_cfg, step3_kwargs)
    s3 = run_step3(
        erased_bgr, nodes, img_w, img_h,
        binary_pre_erase=binary_pre_erase,
        **step3_kwargs,
    )

    _stage("building_graph")
    s4 = run_step4(
        s3.skeleton, s3.branches, s3.endpoints,
        collinear_angle_tol=phase4_cfg["collinear_angle_tol"],
        min_crossing_gap=phase4_cfg["min_crossing_gap"],
        max_crossing_gap=phase4_cfg["max_crossing_gap"],
        elbow_angle_min=phase4_cfg["elbow_angle_min"],
        junction_radius_R=phase4_cfg["junction_radius_R"],
        binary_raw=s3.binary_raw,
        exclude_rects=[],  # no GT background/legend regions for an upload
    )

    SR, Sc = build_sr_sc(nodes, s3.off_page_nodes, s3, s4)
    kept_pairs, _suppressed = veto_short_gap_by_crossing_separation(s3.short_gap_pairs, SR, Sc)
    s3 = dataclasses.replace(s3, short_gap_pairs=kept_pairs)

    s5 = run_step5(nodes, s3.off_page_nodes, s3, s4)
    s6 = run_step6(s5)

    flow_arrows = [n for n in nodes if n.node_type == "flow_arrow" and len(n.ports) == 0]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    run_step7(
        s6.graph, flow_arrows,
        img_gray=img_gray,
        max_assign_dist_px=phase4_cfg["arrow_max_dist_px"],
    )

    _stage("done")
    return s6.graph
