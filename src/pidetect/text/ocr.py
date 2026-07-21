"""Phase 3 — instrument-bubble tag extraction (OCR). See docs/phase3_design.md.

Pipeline, per node with node_type == "instrument":
    1. Crop: node's own bbox + crop_padding_px, white-fill any OTHER instrument node's
       bbox that falls inside the padded region (neighbour-bubble text bleed --
       docs/phase3_design.md §2). Requires the DEDUPED node set (post
       `scored_family_group_key` NMS, see pidetect.graph.erase) -- run_ocr_on_nodes()
       re-asserts this before doing any work, so a cross-subtype duplicate bubble can
       never be OCR'd twice under two different node identities.
    2. Upscale by `upscale_factor` (LANCZOS). No hard binarization -- PaddleOCR gets the
       plain BGR crop.
    3. PaddleOCR (det=True, rec=True) -> per-line (bbox, text, confidence).
    4. Sort returned lines by y-center (top -> bottom).
    5. Parse into (function, loop_number, parse_status, raw_text) via the regex rules
       in configs/phase3.yaml (docs/phase3_design.md §3). Never fabricates: a field is
       populated only when a regex actually validated it.

Public API:
    run_ocr_on_nodes(img_bgr, all_symbol_nodes, cfg) -> None   (mutates in place)
    parse_tag_lines(texts, cfg) -> (function, loop_number, parse_status, raw_text)
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

from pidetect.graph.erase import SymbolNode, assert_no_duplicate_scored_nodes

# Cache of PaddleOCR engine instances, keyed by language -- model loading is expensive
# and must not happen once per bubble.
_ENGINE_CACHE: dict[str, object] = {}


def _get_engine(cfg: dict):
    """Lazily import and construct (once per language) the PaddleOCR engine.

    Deliberately NOT imported at module level: importing pidetect.text.ocr must not
    require paddleocr to be installed (e.g. so parse_tag_lines() stays unit-testable,
    and so the rest of the pipeline doesn't break when Phase 3 isn't set up yet -- see
    the Python-3.14/paddlepaddle compatibility note in requirements.txt).
    """
    lang = cfg.get("ocr_lang", "en")
    if lang not in _ENGINE_CACHE:
        import os
        # paddlex reads this flag once at import time (module-level constant in
        # paddlex.utils.flags), so it must be set before the first `import paddlex`
        # (triggered transitively by `import paddleocr`). Default MKLDNN CPU inference
        # path crashes on this machine's static-graph text-detection model with
        # `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>]` (paddlepaddle 3.3.1 CPU, PIR
        # executor) -- confirmed empirically; disabling MKLDNN avoids that code path.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        from paddleocr import PaddleOCR  # noqa: local import, see docstring above
        # paddleocr>=3.0 renamed use_angle_cls -> use_textline_orientation and dropped
        # show_log (unknown kwargs now raise ValueError instead of being ignored --
        # confirmed empirically against paddleocr 3.7.0/paddlepaddle 3.3.1).
        # use_doc_orientation_classify/use_doc_unwarping default True in paddleocr>=3.0
        # but are dead weight for small, always-upright bubble crops -- disabling both
        # measured a clean ~24% per-call speedup (Phase 5 latency spike) with identical
        # output text.
        _ENGINE_CACHE[lang] = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ENGINE_CACHE[lang]


# ---------------------------------------------------------------------------
# Crop + upscale
# ---------------------------------------------------------------------------

def _build_crop(
    img_bgr: np.ndarray,
    node: SymbolNode,
    all_nodes: list[SymbolNode],
    pad: int,
    mask_third_party: bool,
) -> np.ndarray:
    """Crop node.bbox + pad, white-filling any OTHER instrument node's bbox that
    intersects the padded region (docs/phase3_design.md §2 neighbour-bleed fix)."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(node.x1) - pad)
    y1 = max(0, int(node.y1) - pad)
    x2 = min(w, int(node.x2) + pad)
    y2 = min(h, int(node.y2) + pad)
    crop = img_bgr[y1:y2, x1:x2].copy()

    if mask_third_party:
        for other in all_nodes:
            if other.node_id == node.node_id or other.node_type != "instrument":
                continue
            ox1, oy1, ox2, oy2 = other.x1, other.y1, other.x2, other.y2
            ix1, iy1 = max(ox1, x1), max(oy1, y1)
            ix2, iy2 = min(ox2, x2), min(oy2, y2)
            if ix1 < ix2 and iy1 < iy2:
                crop[int(iy1 - y1):int(iy2 - y1), int(ix1 - x1):int(ix2 - x1)] = 255

    return crop


def _upscale(crop: np.ndarray, factor: int) -> np.ndarray:
    import cv2
    h, w = crop.shape[:2]
    return cv2.resize(crop, (max(1, w * factor), max(1, h * factor)), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# PaddleOCR call
# ---------------------------------------------------------------------------

def _run_paddle_ocr(crop_bgr: np.ndarray, cfg: dict) -> list[tuple[float, str, float]]:
    """Return [(y_center, text, confidence), ...] sorted top -> bottom.

    paddleocr>=3.0's `.predict()` returns a list of `OCRResult` (dict-like) objects,
    one per input image, with parallel `rec_texts` / `rec_scores` / `rec_boxes`
    (`[xmin, ymin, xmax, ymax]`, axis-aligned) lists -- confirmed empirically against
    paddleocr 3.7.0 (the old `[box, (text, conf)]` per-line tuple shape is gone).
    """
    engine = _get_engine(cfg)
    result = engine.predict(crop_bgr)

    lines: list[tuple[float, str, float]] = []
    page = result[0] if result else None
    if not page:
        return lines
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    boxes = page.get("rec_boxes", [])
    for text, score, box in zip(texts, scores, boxes):
        y_center = (float(box[1]) + float(box[3])) / 2
        lines.append((y_center, text, float(score)))

    lines.sort(key=lambda t: t[0])
    return lines


# ---------------------------------------------------------------------------
# Parsing (regex rules, docs/phase3_design.md §3) -- pure function, no PaddleOCR needed
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def parse_tag_lines(texts: list[str], cfg: dict) -> tuple[Optional[str], Optional[str], str, str]:
    """Parse OCR line texts (already sorted top->bottom) into
    (function, loop_number, parse_status, raw_text).

    parse_status in {"ok", "ok_placeholder", "single_line_split",
    "single_line_unsplit", "failed"}. function/loop_number are None unless a regex
    actually validated them -- never fabricated.
    """
    clean = [_WS_RE.sub(" ", t).strip() for t in texts if t and t.strip()]
    raw_text = " ".join(clean)

    func_re = re.compile(cfg["function_regex"])
    loop_re = re.compile(cfg["loop_number_regex"])
    placeholder_re = re.compile(cfg.get("placeholder_regex", r"^X{2,6}$"))
    single_re = re.compile(cfg["single_line_regex"])

    def _loop_status(s: str) -> Optional[str]:
        if loop_re.match(s):
            return "ok"
        if placeholder_re.match(s):
            return "ok_placeholder"
        return None

    if len(clean) >= 2:
        top = clean[0].upper()
        bottom = clean[-1].upper()
        if func_re.match(top):
            status = _loop_status(bottom)
            if status is not None:
                return top, bottom, status, raw_text
        # Recovery: PaddleOCR split into >=2 boxes but they don't validate individually
        # (e.g. a stray fragment box) -- retry via the single-line regex on the
        # whitespace-joined text before giving up.
        m = single_re.match(raw_text.upper())
        if m:
            return m.group(1), m.group(2), "single_line_split", raw_text
        return None, None, "failed", raw_text

    if len(clean) == 1:
        m = single_re.match(clean[0].upper())
        if m:
            return m.group(1), m.group(2), "single_line_split", raw_text
        return None, None, "single_line_unsplit", raw_text

    return None, None, "failed", raw_text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ocr_on_nodes(
    img_bgr: np.ndarray,
    all_symbol_nodes: list[SymbolNode],
    cfg: dict,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Run OCR on every node_type == "instrument" node in `all_symbol_nodes`, mutating
    its tag_* fields in place.

    `all_symbol_nodes` MUST be the DEDUPED node set (post
    centroid_nms(..., group_key=scored_family_group_key) + build_node_set) -- this is
    re-asserted here (not just assumed) so a cross-subtype duplicate bubble can never
    reach the OCR stage under two different node identities and produce two
    conflicting tag records for one physical instrument. Raises AssertionError (from
    pidetect.graph.erase.assert_no_duplicate_scored_nodes) if it isn't.

    `progress_cb`, if given, is called as `progress_cb(i, total)` after each of the
    `total` instrument bubbles is OCR'd (i = 1-indexed count so far) -- OCR dominates
    per-sheet latency (docs/phase5_design.md §2), so this is the one stage worth
    surfacing fine-grained progress for.
    """
    assert_no_duplicate_scored_nodes(all_symbol_nodes, centroid_frac=cfg.get("nms_centroid_frac", 0.5))

    pad = cfg["crop_padding_px"]
    mask_third_party = cfg.get("mask_third_party", True)
    upscale = cfg["upscale_factor"]

    instrument_nodes = [n for n in all_symbol_nodes if n.node_type == "instrument"]
    total = len(instrument_nodes)
    for i, node in enumerate(instrument_nodes, start=1):
        crop = _build_crop(img_bgr, node, all_symbol_nodes, pad, mask_third_party)
        crop = _upscale(crop, upscale)
        lines = _run_paddle_ocr(crop, cfg)
        texts = [t for _, t, _ in lines]
        confs = [c for _, _, c in lines]

        function, loop_number, status, raw_text = parse_tag_lines(texts, cfg)
        node.tag_raw_text = raw_text
        node.tag_function = function
        node.tag_loop_number = loop_number
        node.tag_parse_status = status
        node.tag_confidence = (sum(confs) / len(confs)) if confs else 0.0

        if progress_cb is not None:
            progress_cb(i, total)
