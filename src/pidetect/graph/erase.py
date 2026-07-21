"""Phase 4 steps 0–2: preprocessing, node set, symbol erasure.

Step 0a — ROI filter: drop predictions whose centroid falls inside a GraphML
           `background` bbox.  Falls back to a border-margin mask when no
           background bboxes are available.
Step 0b — Centroid-NMS dedupe: sort by confidence DESC, suppress any lower-confidence
           prediction whose centroid is within nms_centroid_frac × sqrt(pred_w × pred_h)
           of a kept prediction. Grouped by `group_key(cls_id)`, which defaults to
           per-class (cls_id itself) but can map several classes to one shared group --
           see `scored_family_group_key`: a physical valve (14 mutually-exclusive
           subtype classes) or a physical instrument bubble (5 subtype classes) is
           exactly one subtype, so per-class-only NMS never dedupes two overlapping
           predictions of the SAME physical object under two DIFFERENT subtype
           guesses. Confirmed empirically across sheets 0/3/10: 12 bubble-family
           cross-subtype duplicates (docs/phase3_design.md §5, one with 3 overlapping
           predictions on one bubble) and 21 valve-family ones (docs/phase4_final.md
           "Node dedupe correction" -- ~15 pure same-glyph subtype confusion, ~6 where
           one detection spans a valve body + separately-drawn actuator box that
           another detection only partially covers; both merge to one graph node).
           IMPORTANT: this grouping must stay scoped to genuinely-one-object class
           families (valve classes together, instrument classes together). It must
           NEVER be applied globally across unrelated supercategories (e.g. valve vs
           instrument) -- two different real symbols legitimately sitting close
           together (a valve actuator right next to an instrument bubble) must both
           survive as separate nodes; only "these classes are different guesses at
           what THIS SAME glyph is" may be grouped. The 10 remaining cross-class close
           pairs that span node_type (valve/instrument vs unknown_fitting/flow_arrow/
           tag_rect) are explicitly OUT of this grouping and left as a known-open item
           (docs/phase4_final.md).
Step 1  — Node set: tag each surviving prediction as valve / instrument /
           unknown_fitting / flow_arrow / tag_rect.
Step 2  — Erasure: fill all non-flow_arrow symbol bboxes (dilated by
           erase_dilation_px) with the background colour (white).

Public API used by scripts/run_phase4_steps03.py:
    roi_filter(preds, bg_bboxes, img_w, img_h, border_frac) -> list[dict]
    centroid_nms(preds, frac, group_key) -> list[dict]
    scored_family_group_key(cls_id) -> int | str
    build_node_set(preds) -> list[SymbolNode]
    erase_symbols(img, nodes, dilation_px, bg_color) -> np.ndarray
    assert_no_duplicate_scored_nodes(nodes, centroid_frac) -> None
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from pidetect.data.open100 import OUR_ARROW_IDX, OUR_INSTRUMENT_IDX, OUR_VALVE_IDX

# cls indices that are text-region tags: erase but do NOT create graph nodes
_TAG_RECT_IDX: frozenset[int] = frozenset({29, 31})


def scored_family_group_key(cls_id: int) -> "int | str":
    """centroid_nms group_key: group predictions by SCORED supercategory ("valve" or
    "instrument") so a same-object, cross-subtype duplicate within EITHER family is
    deduped -- a physical valve is exactly one of 14 mutually-exclusive subtype
    classes, a physical instrument bubble exactly one of 5. Every other class
    (unknown_fitting, flow_arrow, tag_rect) keeps its own per-class group, unchanged.
    Never merges across supercategories: see module docstring.
    """
    node_type = _classify_pred(cls_id)
    if node_type in ("valve", "instrument"):
        return node_type
    return cls_id


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
    # set by build_node_set when centroid_nms (grouped, e.g. scored_family_group_key)
    # suppressed a same-object prediction of a DIFFERENT class at this location --
    # makes an uncertain subtype call visible instead of silently resolved.
    subtype_conflict: bool = False
    suppressed_subtypes: list[str] = field(default_factory=list)
    # Phase 3 (docs/phase3_design.md) -- set by pidetect.text.ocr.run_ocr_on_nodes() for
    # node_type == "instrument" nodes only. tag_parse_status defaults to "not_run" so a
    # node that never went through OCR is visibly distinct from one OCR gave up on
    # ("failed"). Never fabricated: tag_function/tag_loop_number stay None unless a
    # regex in ocr.py actually validated them.
    tag_raw_text: str = ""
    tag_function: Optional[str] = None
    tag_loop_number: Optional[str] = None
    tag_confidence: float = 0.0
    tag_parse_status: str = "not_run"

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
    group_key: Optional[Callable[[int], "int | str"]] = None,
) -> tuple[list[dict], int]:
    """Centroid-distance NMS, grouped by `group_key(cls_id)` (default: per-class).

    Sort each group by confidence DESC.  Greedily keep the highest-confidence
    prediction; suppress any subsequent same-group prediction whose centroid is
    within centroid_frac × sqrt(pred_w × pred_h) of the kept prediction.

    `group_key` lets several classes share one NMS group -- e.g.
    `scored_family_group_key` merges the 5 instrument_bubble* subtype classes into one
    group and the 14 valve subtype classes into another, since within each family the
    classes are mutually-exclusive guesses at the same kind of physical object and
    per-class-only NMS never dedupes a same-object, different-subtype pair (see module
    docstring). Defaults to `cls_id` itself (original per-class-only behaviour) when
    not given.

    When a suppression crosses a class boundary (the suppressed prediction's cls_id
    differs from the survivor's), the survivor is annotated in-place with
    `subtype_conflict=True` and `suppressed_subtypes` (list of the losing cls_names) --
    so an uncertain subtype call is visible on the surviving node rather than silently
    resolved. Same-class suppression (ordinary duplicate-detection NMS) is not
    annotated; it is not a subtype disagreement.

    Returns (kept_predictions, n_suppressed).
    """
    key_fn = group_key or (lambda cls_id: cls_id)
    by_group: dict = {}
    for pred in predictions:
        by_group.setdefault(key_fn(pred["cls_id"]), []).append(pred)

    kept: list[dict] = []
    n_suppressed = 0
    for group_preds in by_group.values():
        group_preds = sorted(group_preds, key=lambda p: -p["conf"])
        suppressed = [False] * len(group_preds)
        for i, pi in enumerate(group_preds):
            if suppressed[i]:
                continue
            kept.append(pi)
            wi = pi["x2"] - pi["x1"]
            hi = pi["y2"] - pi["y1"]
            thr = centroid_frac * math.sqrt(max(wi * hi, 1.0))
            cxi = (pi["x1"] + pi["x2"]) / 2
            cyi = (pi["y1"] + pi["y2"]) / 2
            for j in range(i + 1, len(group_preds)):
                if suppressed[j]:
                    continue
                pj = group_preds[j]
                cxj = (pj["x1"] + pj["x2"]) / 2
                cyj = (pj["y1"] + pj["y2"]) / 2
                if math.hypot(cxi - cxj, cyi - cyj) < thr:
                    suppressed[j] = True
                    n_suppressed += 1
                    if pj["cls_id"] != pi["cls_id"]:
                        pi["subtype_conflict"] = True
                        pi.setdefault("suppressed_subtypes", []).append(pj["cls_name"])
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
            subtype_conflict=bool(p.get("subtype_conflict", False)),
            suppressed_subtypes=list(p.get("suppressed_subtypes", [])),
        )
        for i, p in enumerate(predictions)
    ]


# ---------------------------------------------------------------------------
# Construction-time regression guard (Phase 3 dedup fix)
# ---------------------------------------------------------------------------

def _assert_no_duplicates_within(
    nodes: list[SymbolNode],
    node_type: str,
    centroid_frac: float,
) -> None:
    group = [n for n in nodes if n.node_type == node_type]
    for i in range(len(group)):
        a = group[i]
        aw, ah = a.x2 - a.x1, a.y2 - a.y1
        thr = centroid_frac * math.sqrt(max(aw * ah, 1.0))
        for j in range(i + 1, len(group)):
            b = group[j]
            d = math.hypot(a.cx - b.cx, a.cy - b.cy)
            if d < thr:
                raise AssertionError(
                    f"Duplicate {node_type} nodes: node {a.node_id} ({a.cls_name}, "
                    f"conf={a.conf:.2f}) and node {b.node_id} ({b.cls_name}, "
                    f"conf={b.conf:.2f}) have centroids {d:.1f}px apart "
                    f"(threshold {thr:.1f}px). scored_family_group_key dedup did not "
                    f"converge -- check centroid_nms is called with "
                    f"group_key=scored_family_group_key before build_node_set()."
                )


def assert_no_duplicate_scored_nodes(
    nodes: list[SymbolNode],
    centroid_frac: float = 0.5,
) -> None:
    """Fail loudly if two `node_type == "instrument"` nodes, OR two
    `node_type == "valve"` nodes, are closer than the standard centroid-NMS radius --
    i.e. the family-grouped dedup (`scored_family_group_key`) did not converge and a
    cross-subtype duplicate reached the node set. Call this AFTER build_node_set(),
    once per sheet, right after centroid_nms has run with
    `group_key=scored_family_group_key`.

    Checks valve and instrument nodes SEPARATELY (never against each other) -- this is
    a regression guard, not the fix: it exists so a future change that bypasses or
    misconfigures the grouped dedup (e.g. reverting to the default per-class-only
    group_key) fails immediately and specifically, instead of silently reintroducing
    duplicate nodes into the connectivity graph.
    """
    _assert_no_duplicates_within(nodes, "instrument", centroid_frac)
    _assert_no_duplicates_within(nodes, "valve", centroid_frac)


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
