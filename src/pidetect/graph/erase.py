"""Phase 4 steps 0–2: preprocessing, node set, symbol erasure.

Step 0a — ROI filter: drop predictions whose centroid falls inside a GraphML
           `background` bbox.  Falls back to a border-margin mask when no
           background bboxes are available.
Step 0b — Centroid-NMS dedupe: per class, sort by confidence DESC, suppress any
           lower-confidence prediction whose centroid is within
           nms_centroid_frac × sqrt(pred_w × pred_h) of a kept prediction.
Step 1  — Node set: tag each surviving prediction as valve / instrument /
           unknown_fitting / flow_arrow / tag_rect.
Step 2  — Erasure: fill all non-flow_arrow symbol bboxes (dilated by
           erase_dilation_px) with the background colour (white).

Public API used by scripts/run_phase4_steps03.py:
    roi_filter(preds, bg_bboxes, img_w, img_h, border_frac) -> list[dict]
    centroid_nms(preds, frac) -> list[dict]
    build_node_set(preds) -> list[SymbolNode]
    erase_symbols(img, nodes, dilation_px, bg_color) -> np.ndarray
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pidetect.data.open100 import OUR_ARROW_IDX, OUR_INSTRUMENT_IDX, OUR_VALVE_IDX

# cls indices that are text-region tags: erase but do NOT create graph nodes
_TAG_RECT_IDX: frozenset[int] = frozenset({29, 31})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SymbolNode:
    """One detected symbol after filtering and type-tagging (step 1).

    node_type values:
        "valve"           — scored, OUR_VALVE_IDX
        "instrument"      — scored, OUR_INSTRUMENT_IDX
        "unknown_fitting" — unscored in-diagram detection (transit node)
        "flow_arrow"      — direction marker, NOT erased, NOT a graph node yet
        "tag_rect"        — text tag bbox, erased only, no graph node
    """
    node_id: int
    cls_id: int
    cls_name: str
    node_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    # filled after erase_symbols()
    dilated: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    # filled during endpoint binding in lines.py
    ports: list[tuple[int, int]] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def is_graph_node(self) -> bool:
        """True if this node should appear in the connectivity graph."""
        return self.node_type != "tag_rect"


# ---------------------------------------------------------------------------
# Step 0a — ROI filter
# ---------------------------------------------------------------------------

def roi_filter(
    predictions: list[dict],
    background_bboxes: Sequence[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    border_margin_frac: float = 0.04,
) -> tuple[list[dict], int]:
    """Drop predictions inside background regions or the border margin.

    Returns (kept_predictions, n_dropped).
    When background_bboxes is non-empty, only those regions are used.
    When empty, fall back to the border-margin heuristic.
    """
    use_bg_boxes = bool(background_bboxes)
    bm = border_margin_frac
    border_x0 = img_w * bm
    border_y0 = img_h * bm
    border_x1 = img_w * (1 - bm)
    border_y1 = img_h * (1 - bm)

    kept, n_dropped = [], 0
    for pred in predictions:
        cx = (pred["x1"] + pred["x2"]) / 2
        cy = (pred["y1"] + pred["y2"]) / 2

        if use_bg_boxes:
            in_bg = any(
                bx1 <= cx <= bx2 and by1 <= cy <= by2
                for bx1, by1, bx2, by2 in background_bboxes
            )
            if in_bg:
                n_dropped += 1
                continue
        else:
            if not (border_x0 <= cx <= border_x1 and border_y0 <= cy <= border_y1):
                n_dropped += 1
                continue

        kept.append(pred)
    return kept, n_dropped


# ---------------------------------------------------------------------------
# Step 0b — Centroid-NMS dedupe
# ---------------------------------------------------------------------------

def centroid_nms(
    predictions: list[dict],
    centroid_frac: float = 0.5,
) -> tuple[list[dict], int]:
    """Per-class centroid-distance NMS.

    Sort each class by confidence DESC.  Greedily keep the highest-confidence
    prediction; suppress any subsequent same-class prediction whose centroid is
    within centroid_frac × sqrt(pred_w × pred_h) of the kept prediction.

    Returns (kept_predictions, n_suppressed).
    """
    by_class: dict[int, list[dict]] = {}
    for pred in predictions:
        by_class.setdefault(pred["cls_id"], []).append(pred)

    kept: list[dict] = []
    n_suppressed = 0
    for cls_preds in by_class.values():
        cls_preds = sorted(cls_preds, key=lambda p: -p["conf"])
        suppressed = [False] * len(cls_preds)
        for i, pi in enumerate(cls_preds):
            if suppressed[i]:
                continue
            kept.append(pi)
            wi = pi["x2"] - pi["x1"]
            hi = pi["y2"] - pi["y1"]
            thr = centroid_frac * math.sqrt(max(wi * hi, 1.0))
            cxi = (pi["x1"] + pi["x2"]) / 2
            cyi = (pi["y1"] + pi["y2"]) / 2
            for j in range(i + 1, len(cls_preds)):
                if suppressed[j]:
                    continue
                pj = cls_preds[j]
                cxj = (pj["x1"] + pj["x2"]) / 2
                cyj = (pj["y1"] + pj["y2"]) / 2
                if math.hypot(cxi - cxj, cyi - cyj) < thr:
                    suppressed[j] = True
                    n_suppressed += 1
    return kept, n_suppressed


# ---------------------------------------------------------------------------
# Step 1 — Node set
# ---------------------------------------------------------------------------

def _classify_pred(cls_id: int) -> str:
    if cls_id in OUR_VALVE_IDX:
        return "valve"
    if cls_id in OUR_INSTRUMENT_IDX:
        return "instrument"
    if cls_id in OUR_ARROW_IDX:
        return "flow_arrow"
    if cls_id in _TAG_RECT_IDX:
        return "tag_rect"
    return "unknown_fitting"


def build_node_set(predictions: list[dict]) -> list[SymbolNode]:
    """Convert filtered+deduped predictions to typed SymbolNode objects.

    All prediction types are included (valve, instrument, unknown_fitting,
    flow_arrow, tag_rect).  Callers filter by node.is_graph_node for graph
    construction; the full list is needed for erasure (step 2).
    """
    return [
        SymbolNode(
            node_id=i,
            cls_id=p["cls_id"],
            cls_name=p["cls_name"],
            node_type=_classify_pred(p["cls_id"]),
            x1=float(p["x1"]),
            y1=float(p["y1"]),
            x2=float(p["x2"]),
            y2=float(p["y2"]),
            conf=p["conf"],
        )
        for i, p in enumerate(predictions)
    ]


# ---------------------------------------------------------------------------
# Step 2 — Erasure
# ---------------------------------------------------------------------------

def _dilate_bbox(
    x1: float, y1: float, x2: float, y2: float,
    margin: int, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, int(x1) - margin),
        max(0, int(y1) - margin),
        min(img_w, int(x2) + margin),
        min(img_h, int(y2) + margin),
    )


def _modal_border_color(img: np.ndarray, margin_px: int = 30) -> int:
    """Estimate background colour from the modal pixel in the border strip."""
    h, w = img.shape[:2]
    gray = img if img.ndim == 2 else np.mean(img, axis=2).astype(np.uint8)
    border = np.concatenate([
        gray[:margin_px, :].ravel(),
        gray[-margin_px:, :].ravel(),
        gray[:, :margin_px].ravel(),
        gray[:, -margin_px:].ravel(),
    ])
    if border.size == 0:
        return 255
    counts = np.bincount(border.astype(np.uint8), minlength=256)
    return int(counts.argmax())


def erase_symbols(
    img: np.ndarray,
    nodes: list[SymbolNode],
    dilation_px: int = 8,
    bg_color: int | None = None,
) -> np.ndarray:
    """Fill all non-flow_arrow node bboxes (dilated by dilation_px) with bg_color.

    Modifies a copy of img.  Also writes node.dilated for all nodes so the
    endpoint-binding step can use the expanded bboxes.

    flow_arrow nodes: dilated field is set to original bbox (no erasure).
    """
    h, w = img.shape[:2]
    erased = img.copy()

    if bg_color is None:
        bg_color = _modal_border_color(img)

    for node in nodes:
        dx1, dy1, dx2, dy2 = _dilate_bbox(
            node.x1, node.y1, node.x2, node.y2, dilation_px, w, h
        )
        node.dilated = (dx1, dy1, dx2, dy2)
        if erased.ndim == 3:
            erased[dy1:dy2, dx1:dx2] = bg_color
        else:
            erased[dy1:dy2, dx1:dx2] = bg_color

    return erased
