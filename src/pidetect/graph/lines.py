"""Phase 4 step 3: binarize → skeleton → branch extraction → endpoint binding.

Pipeline:
    1. binarize()               grayscale → Gaussian blur → Otsu → drawing-frame removal
                                 (mask_drawing_frame) → morphological cleanup
    2. skeletonize()            Zhang-Suen thinning (skimage.morphology.skeletonize)
    3. extract_branches()       skan branch polylines; discard stubs < min_branch_len_px
    4. bind_endpoints()         assign each branch endpoint to the nearest symbol whose
                                dilated bbox (+ extra_bind_px) contains it; contest logging

Public API:
    run_step3(erased_img, nodes, cfg) -> Step3Result
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pidetect.graph.erase import SymbolNode


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    branch_id: int
    src_rc: tuple[int, int]     # (row, col) of src endpoint in image
    dst_rc: tuple[int, int]     # (row, col) of dst endpoint in image
    coords_rc: np.ndarray       # (N, 2) array of (row, col) pixel coords along branch
    length_px: float
    branch_type: int            # 0=tip-tip, 1=tip-junc, 2=junc-junc, 3=cycle

    @property
    def src_xy(self) -> tuple[int, int]:
        """(x, y) of src endpoint."""
        return int(self.src_rc[1]), int(self.src_rc[0])

    @property
    def dst_xy(self) -> tuple[int, int]:
        """(x, y) of dst endpoint."""
        return int(self.dst_rc[1]), int(self.dst_rc[0])


@dataclass
class BoundEndpoint:
    point_xy: tuple[int, int]
    branch_id: int
    which_end: str              # "src" or "dst"
    bound_node_id: Optional[int]  # None = unbound
    contested: bool = False     # True: multiple symbols in range; nearest wins
    is_off_page: bool = False   # True: unbound + within off_page_border_px of sheet edge


@dataclass
class Step3Result:
    skeleton: np.ndarray        # boolean skeleton image
    binary_raw: np.ndarray      # Otsu binary BEFORE morphological close (preserves bridge gaps)
    branches: list[Branch]
    endpoints: list[BoundEndpoint]
    off_page_nodes: list[SymbolNode]  # synthetic nodes for unbound border endpoints
    vessel_nodes: list[SymbolNode] = field(default_factory=list)  # synthetic vessel body nodes
    # (node_id_a, node_id_b) pairs confirmed connected in the pre-erasure binary but whose
    # connecting pipe was fully consumed by overlapping erasure zones — receive a direct edge
    short_gap_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_bound(self) -> int:
        return sum(1 for e in self.endpoints if e.bound_node_id is not None)

    @property
    def n_unbound(self) -> int:
        return sum(1 for e in self.endpoints if e.bound_node_id is None)

    @property
    def n_contests(self) -> int:
        return sum(1 for e in self.endpoints if e.contested)

    @property
    def n_off_page(self) -> int:
        return len(self.off_page_nodes)


# ---------------------------------------------------------------------------
# Step 3a — Binarize
# ---------------------------------------------------------------------------

def mask_drawing_frame(
    binary_bool: np.ndarray,
    search_px: int = 80,
    min_line_frac: float = 0.75,
    clear_margin_px: int = 6,
) -> np.ndarray:
    """Erase the drawing-frame border (+ its ruler ticks) from a binary ink image.

    P&ID sheets are printed inside a rectangular frame drawn close to the image edge,
    with ruler tick marks / row-column labels just outside it. Left untouched, that
    frame skeletonizes into one long, near-continuous branch running most of the
    sheet's width or height -- which can bridge unrelated symbols near opposite edges
    into a spurious "connecting pipe" (verified on OPEN100 sheet 0: a traced edge ran
    the full top border from the corner to an unrelated off-page marker). Real pipes
    never span that close to the border for that long, so this is detectable purely
    from geometry: for each edge, scan the `search_px`-wide band next to it and find
    rows (top/bottom) or columns (left/right) whose ink coverage across the FULL
    opposite dimension exceeds `min_line_frac` -- only a straight border-hugging line
    reaches that; sparse content (text, symbols, short ticks) does not. Measured on
    OPEN100 sheets 0/3/10: real frame lines sit at 0.84-1.0 coverage, everything else
    in the search band is <0.1 -- comfortably separated from `min_line_frac`.

    Clears a `clear_margin_px` band around each detected line (absorbs anti-aliasing
    and the short perpendicular ruler ticks touching it) across the full opposite
    dimension. Does not touch interior content: legitimate pipes and off-page-connector
    stubs run perpendicular to the border and are only trimmed by a few px right at the
    edge, which does not affect off_page_border_px binding (distance-to-edge based, not
    distance-to-frame-line based).

    Returns a new array (does not modify `binary_bool` in place). A no-op (returns a
    copy) if no edge's band contains a qualifying line -- e.g. a border broken up by
    off-page connector arrows/text, as measured on sheet 0's right edge.
    """
    h, w = binary_bool.shape[:2]
    sp = max(1, min(search_px, h // 3, w // 3))
    out = binary_bool.copy()

    top_frac = binary_bool[:sp, :].mean(axis=1)
    bot_frac = binary_bool[h - sp:, :].mean(axis=1)
    left_frac = binary_bool[:, :sp].mean(axis=0)
    right_frac = binary_bool[:, w - sp:].mean(axis=0)

    for i, frac in enumerate(top_frac):
        if frac >= min_line_frac:
            lo, hi = max(0, i - clear_margin_px), min(sp, i + clear_margin_px + 1)
            out[lo:hi, :] = False
    for j, frac in enumerate(bot_frac):
        if frac >= min_line_frac:
            row = h - sp + j
            lo, hi = max(h - sp, row - clear_margin_px), min(h, row + clear_margin_px + 1)
            out[lo:hi, :] = False
    for i, frac in enumerate(left_frac):
        if frac >= min_line_frac:
            lo, hi = max(0, i - clear_margin_px), min(sp, i + clear_margin_px + 1)
            out[:, lo:hi] = False
    for j, frac in enumerate(right_frac):
        if frac >= min_line_frac:
            col = w - sp + j
            lo, hi = max(w - sp, col - clear_margin_px), min(w, col + clear_margin_px + 1)
            out[:, lo:hi] = False

    return out


def binarize(
    img: np.ndarray,
    blur_sigma: float = 0.8,
    close_disk_r: int = 2,
    open_disk_r: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale → Gaussian blur → Otsu threshold → frame removal → morphological cleanup.

    Returns (binary_final, binary_raw) where:
      binary_final — post-close/open boolean array used for skeletonization
      binary_raw   — post-Otsu, post-frame-removal boolean array BEFORE closing
                     (preserves bridge gaps)

    Frame removal (mask_drawing_frame) runs on every call -- both the erased-image pass
    (skeletonization) and the pre-erasure pass (short-gap corridor ink checks, junction
    crossing-gap probing via Step3Result.binary_raw) share this one choke point, so the
    drawing-frame border can never leak into ANY downstream connectivity mechanism.
    """
    from skimage.morphology import closing, opening, disk

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Gaussian blur: kernel must be odd and positive
    k = max(1, int(2 * math.ceil(2 * blur_sigma) + 1))
    k = k if k % 2 == 1 else k + 1
    blurred = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    # Otsu threshold (invert so lines=True)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_raw = mask_drawing_frame(binary.astype(bool))   # before morphological ops
    binary_bool = binary_raw.copy()

    if close_disk_r > 0:
        binary_bool = closing(binary_bool, disk(close_disk_r))
    if open_disk_r > 0:
        binary_bool = opening(binary_bool, disk(open_disk_r))

    return binary_bool, binary_raw


# ---------------------------------------------------------------------------
# Step 3b — Skeletonize
# ---------------------------------------------------------------------------

def skeletonize_image(binary_bool: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning → 1-pixel-wide boolean skeleton."""
    from skimage.morphology import skeletonize
    return skeletonize(binary_bool)


# ---------------------------------------------------------------------------
# Step 3c — Branch extraction
# ---------------------------------------------------------------------------

def _branches_from_skan(
    skeleton_bool: np.ndarray,
    min_len_px: float,
) -> list[Branch]:
    try:
        from skan import Skeleton, summarize
    except ImportError as exc:
        raise ImportError(
            "skan is required for branch extraction: pip install skan"
        ) from exc

    if not skeleton_bool.any():
        return []

    skel_obj = Skeleton(skeleton_bool, keep_images=True)
    df = summarize(skel_obj, separator="-")

    # Normalise column names across skan versions
    col_map: dict[str, str] = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "-")
        col_map[col] = lc
    df = df.rename(columns=col_map)

    branches: list[Branch] = []
    for idx, row in df.iterrows():
        length = float(row.get("branch-distance", 0.0))
        if length < min_len_px:
            continue

        btype = int(row.get("branch-type", -1))

        # Endpoint coordinates — try image-space first, fall back to coord-space
        def _get_rc(prefix: str) -> tuple[int, int]:
            r_key = f"image-coord-{prefix}-0"
            c_key = f"image-coord-{prefix}-1"
            if r_key in row.index and c_key in row.index:
                return int(row[r_key]), int(row[c_key])
            # older skan: coord-src-0 is already in image space
            r_key2 = f"coord-{prefix}-0"
            c_key2 = f"coord-{prefix}-1"
            if r_key2 in row.index and c_key2 in row.index:
                return int(row[r_key2]), int(row[c_key2])
            return (0, 0)

        src_rc = _get_rc("src")
        dst_rc = _get_rc("dst")

        try:
            coords = skel_obj.path_coordinates(int(idx))
        except Exception:
            coords = np.array([list(src_rc), list(dst_rc)])

        branches.append(Branch(
            branch_id=int(idx),
            src_rc=src_rc,
            dst_rc=dst_rc,
            coords_rc=np.asarray(coords, dtype=np.int32),
            length_px=length,
            branch_type=btype,
        ))
    return branches


def extract_branches(
    skeleton_bool: np.ndarray,
    min_branch_len_px: float = 20.0,
) -> list[Branch]:
    """Extract branch polylines from the skeleton via skan."""
    return _branches_from_skan(skeleton_bool, min_branch_len_px)


# ---------------------------------------------------------------------------
# Step 3d — Endpoint binding
# ---------------------------------------------------------------------------

def _in_dilated_box(
    px: int, py: int,
    dx1: int, dy1: int, dx2: int, dy2: int,
    extra: int,
) -> bool:
    return (dx1 - extra <= px <= dx2 + extra and
            dy1 - extra <= py <= dy2 + extra)


def _near_border(px: int, py: int, img_w: int, img_h: int, margin: int) -> bool:
    return (px < margin or px >= img_w - margin or
            py < margin or py >= img_h - margin)


def bind_endpoints(
    branches: list[Branch],
    nodes: list[SymbolNode],
    img_w: int,
    img_h: int,
    extra_bind_px: int = 5,
    off_page_border_px: int = 30,
) -> tuple[list[BoundEndpoint], list[SymbolNode]]:
    """Assign branch endpoints to symbol nodes.

    For each endpoint:
    1. Collect all graph nodes (non-flow_arrow, non-tag_rect) whose
       dilated_bbox expanded by extra_bind_px contains the endpoint.
    2. If exactly one: bind to it; add the endpoint to node.ports.
    3. If more than one: contest — bind to nearest by centroid distance, log contest.
    4. If none: check if near the sheet border → create an off_page node.

    Returns (bound_endpoints, off_page_nodes).
    """
    graph_nodes = [n for n in nodes if n.is_graph_node]

    bound_endpoints: list[BoundEndpoint] = []
    off_page_nodes: list[SymbolNode] = []
    # start off_page ids after existing nodes
    next_op_id = max((n.node_id for n in nodes), default=-1) + 1

    # deduplicate off_page locations so near-border endpoints on the same branch
    # don't create two redundant off_page nodes
    off_page_seen: set[tuple[int, int]] = set()

    for branch in branches:
        for which_end in ("src", "dst"):
            rc = branch.src_rc if which_end == "src" else branch.dst_rc
            py_img, px_img = int(rc[0]), int(rc[1])  # row=y, col=x

            candidates: list[tuple[float, float, SymbolNode]] = []
            for node in graph_nodes:
                dx1, dy1, dx2, dy2 = node.dilated
                if _in_dilated_box(px_img, py_img, dx1, dy1, dx2, dy2, extra_bind_px):
                    # Primary: distance from endpoint to dilated-bbox edge (0 if inside).
                    # This correctly favours large vessel nodes whose edges are close to
                    # the pipe stub over smaller YOLO nodes with a closer centroid.
                    bex = max(dx1 - px_img, 0, px_img - dx2)
                    bey = max(dy1 - py_img, 0, py_img - dy2)
                    edge_d = math.hypot(bex, bey)
                    ctr_d = math.hypot(px_img - node.cx, py_img - node.cy)
                    candidates.append((edge_d, ctr_d, node))

            if candidates:
                contested = len(candidates) > 1
                candidates.sort(key=lambda t: (t[0], t[1]))
                winner = candidates[0][2]
                winner.ports.append((px_img, py_img))
                bound_endpoints.append(BoundEndpoint(
                    point_xy=(px_img, py_img),
                    branch_id=branch.branch_id,
                    which_end=which_end,
                    bound_node_id=winner.node_id,
                    contested=contested,
                ))
            else:
                is_op = _near_border(px_img, py_img, img_w, img_h, off_page_border_px)
                if is_op:
                    # deduplicate: snap to 20px grid to merge nearby border endpoints
                    key = (round(px_img / 20) * 20, round(py_img / 20) * 20)
                    if key not in off_page_seen:
                        off_page_seen.add(key)
                        op_node = SymbolNode(
                            node_id=next_op_id,
                            cls_id=-1,
                            cls_name="off_page",
                            node_type="off_page",
                            x1=float(px_img - 4),
                            y1=float(py_img - 4),
                            x2=float(px_img + 4),
                            y2=float(py_img + 4),
                            conf=1.0,
                            dilated=(px_img - 4, py_img - 4, px_img + 4, py_img + 4),
                        )
                        op_node.ports.append((px_img, py_img))
                        off_page_nodes.append(op_node)
                        next_op_id += 1
                    # find the off_page node just created (or already present)
                    op_id = None
                    for op in off_page_nodes:
                        if (round(op.cx / 20) * 20, round(op.cy / 20) * 20) == key:
                            op_id = op.node_id
                            break
                else:
                    op_id = None

                bound_endpoints.append(BoundEndpoint(
                    point_xy=(px_img, py_img),
                    branch_id=branch.branch_id,
                    which_end=which_end,
                    bound_node_id=op_id,
                    contested=False,
                    is_off_page=is_op,
                ))

    return bound_endpoints, off_page_nodes


# ---------------------------------------------------------------------------
# Step 3e — Short-gap bridging
# ---------------------------------------------------------------------------
# When two symbol dilated-bboxes overlap or are within short_gap_max_px of each other,
# the erasure pass may have consumed the entire pipe segment between them.  We recover
# these connections by checking the PRE-ERASURE binarized image.
#
# Design decision: if dark (line) pixels existed between the two symbol footprints in the
# original image, we create a direct graph edge even though the skeleton produced no branch.
# This is preferred over shrinking the dilation margin, which would reintroduce skeleton
# noise from symbol-body pixels.  See docs/phase4_design.md §3 for rationale.

def _dilated_gap(a: SymbolNode, b: SymbolNode) -> float:
    """Euclidean gap between the dilated bboxes of two nodes (0.0 when overlapping)."""
    ax1, ay1, ax2, ay2 = a.dilated
    bx1, by1, bx2, by2 = b.dilated
    gx = max(0, ax1 - bx2, bx1 - ax2)
    gy = max(0, ay1 - by2, by1 - ay2)
    return math.hypot(gx, gy)


def _has_pre_erasure_path(
    binary: np.ndarray,   # dark=True, pre-erasure Otsu binary
    a: SymbolNode,
    b: SymbolNode,
) -> bool:
    """True if dark (line) pixels exist between the original bboxes of a and b.

    Crops the combined bbox region, masks out both symbol footprints, and checks for any
    remaining dark pixels in the inter-symbol gap.  When the original bboxes overlap
    (gap ≤ 0) the symbols are touching and assumed connected.
    """
    # Check whether original (un-dilated) bboxes overlap
    orig_gap_x = max(0.0, a.x1 - b.x2, b.x1 - a.x2)
    orig_gap_y = max(0.0, a.y1 - b.y2, b.y1 - a.y2)
    if orig_gap_x == 0.0 and orig_gap_y == 0.0:
        return True  # bboxes touch or overlap — pipe definitely present

    H, W = binary.shape
    rx1 = max(0, int(min(a.x1, b.x1)) - 2)
    ry1 = max(0, int(min(a.y1, b.y1)) - 2)
    rx2 = min(W, int(max(a.x2, b.x2)) + 2)
    ry2 = min(H, int(max(a.y2, b.y2)) + 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return False

    crop = binary[ry1:ry2, rx1:rx2].copy()

    # Mask out both symbol original footprints (their pixels are symbol glyphs, not pipe)
    for node in (a, b):
        nx1 = max(0, int(node.x1) - rx1)
        ny1 = max(0, int(node.y1) - ry1)
        nx2 = min(crop.shape[1], int(node.x2) - rx1)
        ny2 = min(crop.shape[0], int(node.y2) - ry1)
        if nx1 < nx2 and ny1 < ny2:
            crop[ny1:ny2, nx1:nx2] = False

    return bool(crop.any())


def _stub_direction_xy(branch: Branch, which_end: str, k: int = 5) -> tuple[float, float]:
    """Unit vector from a bound endpoint along the branch, looking k pixels inward.

    Returns (dx, dy) normalised to unit length, or (0, 0) if the branch is degenerate.
    Coords are in (row, col) = (y, x) order; we return (col_diff, row_diff) = (dx, dy).
    """
    coords = branch.coords_rc
    n = len(coords)
    if n < 2:
        return (0.0, 0.0)
    if which_end == "src":
        far = min(k, n - 1)
        dr = int(coords[far][0]) - int(coords[0][0])
        dc = int(coords[far][1]) - int(coords[0][1])
    else:  # "dst"
        far = max(n - 1 - k, 0)
        dr = int(coords[far][0]) - int(coords[-1][0])
        dc = int(coords[far][1]) - int(coords[-1][1])
    mag = math.hypot(dc, dr)   # col=x, row=y
    if mag < 0.5:
        return (0.0, 0.0)
    return (dc / mag, dr / mag)


def _node_stub_classify(
    a: SymbolNode,
    b: SymbolNode,
    branches_by_id: dict[int, "Branch"],
    eps_by_node: dict[int, list["BoundEndpoint"]],
    angle_tol_deg: float,
    k: int = 5,
) -> Optional[str]:
    """Classify A's existing (post-erasure skeleton) stub(s) relative to the A->B
    direction into one of three buckets, or None when there is no evidence at all.

    A multi-port symbol (e.g. a valve with an inlet AND an outlet) can have a bound
    stub belonging to a completely different, unrelated connection -- that stub
    pointing away from B is NOT evidence against an A-B edge, it is evidence FOR a
    different edge. So the three buckets are:

      "toward"       -- some stub is within angle_tol_deg of pointing at B: direct
                        positive evidence for the A-B connection.
      "perpendicular"-- no stub points toward B, but at least one is roughly
                        perpendicular (between angle_tol_deg and 180-angle_tol_deg
                        of the A->B direction): the shared-trunk-tap signature.
      "away"         -- every stub points roughly opposite B (within angle_tol_deg
                        of 180 deg off): most consistent with an unrelated port on
                        A's far side, not with tapping a trunk toward B. Treated the
                        same as "no evidence" by the caller.

    Returns None if A has no bound endpoints, or none with a usable direction.
    """
    eps = eps_by_node.get(a.node_id, [])
    if not eps:
        return None

    to_bx = b.cx - a.cx
    to_by = b.cy - a.cy
    to_b_len = math.hypot(to_bx, to_by)
    if to_b_len < 1.0:
        return None
    to_bx /= to_b_len
    to_by /= to_b_len
    cos_toward = math.cos(math.radians(angle_tol_deg))
    cos_away = math.cos(math.radians(180.0 - angle_tol_deg))  # negative

    saw_direction = False
    saw_perpendicular = False
    for ep in eps:
        br = branches_by_id.get(ep.branch_id)
        if br is None:
            continue
        sdx, sdy = _stub_direction_xy(br, ep.which_end, k)
        if sdx == 0.0 and sdy == 0.0:
            continue
        saw_direction = True
        dot = sdx * to_bx + sdy * to_by
        if dot >= cos_toward:
            return "toward"
        if dot > cos_away:
            saw_perpendicular = True
    if not saw_direction:
        return None
    return "perpendicular" if saw_perpendicular else "away"


def _stub_direction_ok(
    a: SymbolNode,
    b: SymbolNode,
    branches_by_id: dict[int, "Branch"],
    eps_by_node: dict[int, list["BoundEndpoint"]],
    angle_tol_deg: float,
    k: int = 5,
) -> bool:
    """Shared-header discriminator: reject a short-gap pair only when NEITHER node
    shows any stub pointing toward the other (no positive evidence anywhere) AND at
    least one node shows a roughly-perpendicular stub (the signature of an
    independent tap onto a shared trunk parallel to the A-B axis). A node with no
    stub at all, or whose stub(s) point roughly opposite B (a likely unrelated port
    on its far side), is never treated as evidence against the pair -- only a clear
    perpendicular reading counts, and only when nothing else points toward.
    """
    a_class = _node_stub_classify(a, b, branches_by_id, eps_by_node, angle_tol_deg, k)
    b_class = _node_stub_classify(b, a, branches_by_id, eps_by_node, angle_tol_deg, k)
    if a_class == "toward" or b_class == "toward":
        return True
    if a_class == "perpendicular" or b_class == "perpendicular":
        return False
    return True


def _has_no_interposing_node(
    a: SymbolNode,
    b: SymbolNode,
    graph_nodes: list[SymbolNode],
    interpose_tol_px: float,
) -> bool:
    """True when no third graph node interposes between A and B on segment A→B.

    A node C interposes when:
      - Its centroid projects onto segment A→B at parameter t ∈ (0.05, 0.95)
        (strictly between A and B, not at the endpoints themselves).
      - The perpendicular distance from C's centroid to line A→B ≤ interpose_tol_px.

    This rejects long-range pairs where a nearer symbol sits on the same pipe run
    (e.g. a row of collinear valves A→C→B where C sits between A and B: the pair
    (A, B) is rejected; (A, C) and (C, B) are accepted because neither has a node
    that falls between them on their respective shorter segments).
    """
    ax, ay = a.cx, a.cy
    bx, by = b.cx, b.cy
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1.0:
        return True  # degenerate: A and B at same point

    seg_len = math.sqrt(seg_len_sq)
    for c in graph_nodes:
        if c.node_id == a.node_id or c.node_id == b.node_id:
            continue
        cx, cy = c.cx, c.cy
        t = ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq
        if not (0.15 <= t <= 0.85):
            continue  # C too close to an endpoint — not meaningfully between A and B
        perp = abs((cy - ay) * dx - (cx - ax) * dy) / seg_len
        if perp <= interpose_tol_px:
            return False  # C lies on the A→B pipe run — reject (A, B)

    return True


def _masked_crop(
    binary: np.ndarray,
    a: SymbolNode,
    b: SymbolNode,
    third_party_nodes: Optional[list[SymbolNode]] = None,
    margin: int = 2,
) -> tuple[np.ndarray, int, int]:
    """Crop the A-B bounding region and clear A's/B's own original bboxes (+ third-party
    dilated bboxes when given). Returns (crop, rx1, ry1) so callers can map crop-local
    pixel coords back to image space. Shared by _corridor_ink_check and _has_touch_ink.
    """
    H, W = binary.shape
    rx1 = max(0, int(min(a.x1, b.x1)) - margin)
    ry1 = max(0, int(min(a.y1, b.y1)) - margin)
    rx2 = min(W, int(max(a.x2, b.x2)) + margin)
    ry2 = min(H, int(max(a.y2, b.y2)) + margin)
    if rx2 <= rx1 or ry2 <= ry1:
        return np.zeros((0, 0), dtype=bool), rx1, ry1

    crop = binary[ry1:ry2, rx1:rx2].copy()

    for node in (a, b):
        nx1 = max(0, int(node.x1) - rx1)
        ny1 = max(0, int(node.y1) - ry1)
        nx2 = min(crop.shape[1], int(node.x2) - rx1)
        ny2 = min(crop.shape[0], int(node.y2) - ry1)
        if nx1 < nx2 and ny1 < ny2:
            crop[ny1:ny2, nx1:nx2] = False

    if third_party_nodes:
        for node in third_party_nodes:
            if node.node_id in (a.node_id, b.node_id):
                continue
            dx1, dy1, dx2, dy2 = node.dilated
            nx1 = max(0, int(dx1) - rx1)
            ny1 = max(0, int(dy1) - ry1)
            nx2 = min(crop.shape[1], int(dx2) - rx1)
            ny2 = min(crop.shape[0], int(dy2) - ry1)
            if nx1 < nx2 and ny1 < ny2:
                crop[ny1:ny2, nx1:nx2] = False

    return crop, rx1, ry1


_TOUCH_INK_MARGIN_PX = 10  # wider than the 2px corridor margin: touching bboxes leave
                           # no along-axis gap to search, so a genuine connecting mark
                           # for these pairs is more often just outside the tight bbox
                           # union than for a normal-gap pair (verified against sheet-0/10
                           # TPs sym_18<->168, sym_24<->185, which need 8-10px to surface
                           # 2-14px of real ink; confirmed-absent pairs stay at 0px even here)


def _has_touch_ink(
    binary: np.ndarray,
    a: SymbolNode,
    b: SymbolNode,
    third_party_nodes: Optional[list[SymbolNode]] = None,
) -> bool:
    """Minimal ink check for the bbox-touch/overlap case (no along-axis gap to measure
    continuity over, so the full corridor test doesn't apply). Requires at least one
    dark pixel left in the A-B region (searched with a wider margin than the standard
    corridor crop, see _TOUCH_INK_MARGIN_PX) after clearing A's/B's own bodies (and any
    third-party dilated footprints) -- i.e. some ink beyond "the two symbols happen to
    be adjacent" with zero evidence of an actual connecting mark between them.
    """
    crop, _, _ = _masked_crop(binary, a, b, third_party_nodes, margin=_TOUCH_INK_MARGIN_PX)
    return bool(crop.any())


def _corridor_ink_check(
    binary: np.ndarray,
    a: SymbolNode,
    b: SymbolNode,
    corridor_half_width: float,
    continuity_frac: float,
    perp_reject_ratio: float,
    third_party_nodes: Optional[list[SymbolNode]] = None,
) -> bool:
    """Full corridor test on a single binary array (A/B original bboxes masked inside).

    Returns True when dark pixels in the corridor form an aligned, continuous stripe
    that is not dominated by perpendicular content.  Called by _has_aligned_corridor_path
    for both the primary (masked) and fallback (unmasked) binaries.

    `third_party_nodes`, when given, are nodes OTHER than a/b whose DILATED bboxes get
    cleared from the crop too (mask_all_nodes behaviour) -- this excludes neighbour-symbol
    body ink from being misread as connecting corridor ink.  Deliberately excludes a/b
    themselves: only their ORIGINAL (undilated) footprints are cleared below, so a genuine
    connecting stub that lives in a's or b's own dilation margin (a common case for
    close-together pairs) is never erased by this test.
    """
    crop, rx1, ry1 = _masked_crop(binary, a, b, third_party_nodes)
    if crop.size == 0:
        return False

    if not crop.any():
        return False

    rows, cols = np.where(crop)
    px = cols.astype(float) + rx1
    py = rows.astype(float) + ry1

    dx = b.cx - a.cx
    dy = b.cy - a.cy
    seg_len = math.hypot(dx, dy)
    if seg_len < 1.0:
        return True

    ux, uy = dx / seg_len, dy / seg_len
    along = (px - a.cx) * ux + (py - a.cy) * uy
    perp_abs = np.abs((px - a.cx) * (-uy) + (py - a.cy) * ux)

    in_corridor = perp_abs <= corridor_half_width
    along_corr = along[in_corridor]

    if len(along_corr) == 0:
        return False

    gap_len = float(along_corr.max() - along_corr.min())
    if gap_len < 2.0:
        return True  # tiny gap, corridor fully spanned

    bin_size = 2.0
    n_bins = max(1, int(gap_len / bin_size))
    bins_occ = len(
        np.unique(np.floor((along_corr - along_corr.min()) / bin_size).astype(np.int32))
    )
    if bins_occ / n_bins < continuity_frac:
        return False

    along_lo, along_hi = float(along_corr.min()), float(along_corr.max())
    in_range = (along >= along_lo) & (along <= along_hi)
    n_corr  = int(np.sum(in_corridor & in_range))
    n_wider = int(np.sum(
        (perp_abs > corridor_half_width) & (perp_abs <= corridor_half_width * 3.0) & in_range
    ))
    if n_corr > 0 and n_wider > n_corr * perp_reject_ratio:
        return False

    return True


def _has_aligned_corridor_path(
    binary: np.ndarray,
    a: SymbolNode,
    b: SymbolNode,
    corridor_half_width: float = 8.0,
    continuity_frac: float = 0.4,
    perp_reject_ratio: float = 2.0,
    binary_unmasked: Optional[np.ndarray] = None,
    third_party_nodes: Optional[list[SymbolNode]] = None,
) -> bool:
    """Pre-erasure corridor check: dark pixels must form an aligned stripe along A→B.

    Primary test runs on `binary` (the raw pre-erasure binary) with every OTHER node's
    dilated bbox cleared via `third_party_nodes` when mask_all_nodes is active -- this
    excludes neighbour-symbol-body ink without erasing A's/B's own dilation margin,
    where a genuine connecting stub commonly lives for close-together pairs.  If the
    primary test fails and `binary_unmasked` is provided, the SAME full corridor test
    (continuity + perp-rejection, no third-party masking) is re-run on the unmasked
    binary as a fallback.

    Original-bbox overlap case: when the symbol footprints already touch or overlap,
    there is no along-axis gap left to run the full continuity/perp-reject corridor
    test over (gap_len ~ 0) -- but this is NOT an unconditional pass. It requires the
    minimal ink check (_has_touch_ink): at least one dark pixel beyond A's/B's own
    bodies (and third-party dilated footprints) must be present, i.e. some genuine
    mark between them, not just spatial adjacency with zero connecting evidence.
    """
    gx = max(0.0, a.x1 - b.x2, b.x1 - a.x2)
    gy = max(0.0, a.y1 - b.y2, b.y1 - a.y2)
    if gx == 0.0 and gy == 0.0:
        if _has_touch_ink(binary, a, b, third_party_nodes):
            return True
        if binary_unmasked is not None and _has_touch_ink(binary_unmasked, a, b, None):
            return True
        return False

    if _corridor_ink_check(binary, a, b, corridor_half_width, continuity_frac, perp_reject_ratio,
                           third_party_nodes):
        return True

    if binary_unmasked is not None:
        return _corridor_ink_check(
            binary_unmasked, a, b, corridor_half_width, continuity_frac, perp_reject_ratio,
        )

    return False


def find_short_gap_pairs(
    nodes: list[SymbolNode],
    binary_pre_erase: np.ndarray,
    short_gap_max_px: float = 50.0,
    interpose_tol_px: float = 20.0,
    corridor_half_width_px: float = 10.0,
    continuity_frac: float = 0.5,
    perp_reject_ratio: float = 3.0,
    binary_unmasked: Optional[np.ndarray] = None,
    mask_third_party_nodes: Optional[list[SymbolNode]] = None,
    branches: Optional[list["Branch"]] = None,
    endpoints: Optional[list["BoundEndpoint"]] = None,
    stub_angle_tol_deg: Optional[float] = None,
) -> list[tuple[int, int]]:
    """Find (node_id_a, node_id_b) pairs whose connecting pipe was fully erased.

    A pair qualifies when ALL of these hold:
      1. The gap between their dilated bboxes is <= short_gap_max_px.
      2. No third graph node interposes on the A→B segment (directness filter).
      3. The pre-erasure binary shows an aligned stripe of dark pixels along A→B
         (corridor continuity + perpendicular-dominance rejection).
      4. (when stub_angle_tol_deg is given) Neither A nor B has an EXISTING
         post-erasure skeleton stub that points away from the other node -- see
         _stub_direction_ok. Nodes with no surviving stub at all (fully erased) are
         not penalised by this condition.

    Only graph nodes (is_graph_node=True) are considered.

    When `binary_unmasked` is provided, _has_aligned_corridor_path will fall back to
    testing the unmasked binary if the primary (masked) binary test fails.  Pass the
    original pre-erasure binary here when mask_all_nodes=True in run_step3.

    When `mask_third_party_nodes` is given (mask_all_nodes=True in run_step3), every
    node in that list OTHER than the current pair (a, b) has its DILATED bbox cleared
    from the per-pair corridor crop -- excluding neighbour-symbol-body ink -- while a's
    and b's own dilation margins (where a genuine connecting stub commonly lives) are
    left intact. This masking is computed per-pair (cheap: only rectangles intersecting
    the small A-B crop matter), not as one shared whole-image binary.

    When `stub_angle_tol_deg` is given (requires `branches` and `endpoints` too), the
    shared-header discriminator is active: a pair is rejected if A or B already has a
    bound skeleton stub whose direction clearly does NOT point toward the other node
    (evidence both are independently, perpendicularly tapping a shared trunk rather
    than being directly connected to each other).
    """
    graph_nodes = [n for n in nodes if n.is_graph_node]
    pairs: list[tuple[int, int]] = []

    use_stub_test = stub_angle_tol_deg is not None and branches is not None and endpoints is not None
    if use_stub_test:
        branches_by_id = {b.branch_id: b for b in branches}
        eps_by_node: dict[int, list["BoundEndpoint"]] = {}
        for ep in endpoints:
            if ep.bound_node_id is not None:
                eps_by_node.setdefault(ep.bound_node_id, []).append(ep)

    for i in range(len(graph_nodes)):
        for j in range(i + 1, len(graph_nodes)):
            a, b = graph_nodes[i], graph_nodes[j]
            if _dilated_gap(a, b) > short_gap_max_px:
                continue
            if not _has_no_interposing_node(a, b, graph_nodes, interpose_tol_px):
                continue
            if not _has_aligned_corridor_path(
                binary_pre_erase, a, b,
                corridor_half_width_px, continuity_frac, perp_reject_ratio,
                binary_unmasked,
                third_party_nodes=mask_third_party_nodes,
            ):
                continue
            if use_stub_test and not _stub_direction_ok(
                a, b, branches_by_id, eps_by_node, stub_angle_tol_deg
            ):
                continue
            pairs.append((a.node_id, b.node_id))

    return pairs


# ---------------------------------------------------------------------------
# Step 3 combined entry point
# ---------------------------------------------------------------------------

def run_step3(
    erased_img: np.ndarray,
    nodes: list[SymbolNode],
    img_w: int,
    img_h: int,
    blur_sigma: float = 0.8,
    close_disk_r: int = 2,
    open_disk_r: int = 1,
    min_branch_len_px: float = 20.0,
    endpoint_bind_extra_px: int = 5,
    off_page_border_px: int = 30,
    binary_pre_erase: Optional[np.ndarray] = None,
    mask_all_nodes: bool = False,
    mask_all_nodes_fallback: bool = True,
    short_gap_max_px: float = 50.0,
    interpose_tol_px: float = 20.0,
    corridor_half_width_px: float = 10.0,
    continuity_frac: float = 0.5,
    perp_reject_ratio: float = 3.0,
    stub_angle_tol_deg: Optional[float] = None,
) -> Step3Result:
    """Run the full step-3 pipeline on a symbol-erased image.

    `nodes` should already include any GT-derived vessel nodes (tank/pump).

    When `binary_pre_erase` is not None, short-gap bridging is enabled: pairs of
    nearby symbols whose connecting pipe was fully consumed by erasure are detected via
    the directional corridor check on the pre-erasure binary image.

    `mask_all_nodes=True`: for each candidate short-gap pair (A, B), clear the DILATED
    bboxes of every OTHER node (not A and B themselves) from that pair's corridor crop
    before the ink check.  This prevents third-symbol body pixels from masquerading as
    connecting-pipe ink, while leaving A's and B's own dilation margins intact -- a
    genuine connecting stub commonly lives there for close-together pairs, so clearing
    it (as an earlier whole-image version of this masking did) would wrongly erase real
    evidence for exactly the pairs this mechanism is supposed to recover.

    `mask_all_nodes_fallback=True` (default, only meaningful when mask_all_nodes=True):
    also pass the original unmasked binary to find_short_gap_pairs so that pairs whose
    genuine connecting-pipe ink was removed by masking can be recovered via the same full
    corridor test (continuity + perp-rejection) on the unmasked binary.  Set to False
    for ablation measurements that isolate the masking effect without fallback recovery.

    `stub_angle_tol_deg`, when given, activates the shared-header discriminator: a
    short-gap pair is rejected if A or B already has a surviving post-erasure skeleton
    stub whose direction does NOT point toward the other node within this many degrees
    -- evidence both are independently, perpendicularly tapping a shared trunk parallel
    to the A-B axis rather than being directly connected. Nodes with no surviving stub
    at all (the fully-erased population) are not penalised; this condition only fires
    where stub evidence actually exists.

    Returns a Step3Result with skeleton, branches, bound endpoints, off_page nodes,
    and (when binary_pre_erase is given) short_gap_pairs.
    """
    binary, binary_raw = binarize(erased_img, blur_sigma, close_disk_r, open_disk_r)
    skeleton = skeletonize_image(binary)
    branches = extract_branches(skeleton, min_branch_len_px)
    endpoints, off_page_nodes = bind_endpoints(
        branches, nodes, img_w, img_h,
        extra_bind_px=endpoint_bind_extra_px,
        off_page_border_px=off_page_border_px,
    )
    short_gap_pairs: list[tuple[int, int]] = []
    if binary_pre_erase is not None:
        if mask_all_nodes:
            mask_third_party_nodes = [
                n for n in nodes if getattr(n, "node_type", "") != "flow_arrow"
            ]
            binary_unmasked: Optional[np.ndarray] = (
                binary_pre_erase if mask_all_nodes_fallback else None
            )
        else:
            mask_third_party_nodes = None
            binary_unmasked = None
        short_gap_pairs = find_short_gap_pairs(
            nodes, binary_pre_erase,
            short_gap_max_px, interpose_tol_px,
            corridor_half_width_px, continuity_frac, perp_reject_ratio,
            binary_unmasked,
            mask_third_party_nodes=mask_third_party_nodes,
            branches=branches,
            endpoints=endpoints,
            stub_angle_tol_deg=stub_angle_tol_deg,
        )
    return Step3Result(
        skeleton=skeleton,
        binary_raw=binary_raw,
        branches=branches,
        endpoints=endpoints,
        off_page_nodes=off_page_nodes,
        short_gap_pairs=short_gap_pairs,
    )


# ---------------------------------------------------------------------------
# Config-wiring assertion
# ---------------------------------------------------------------------------
# Root-cause note (docs/phase4_final.md Task 2): mask_all_nodes/mask_all_nodes_fallback
# were defined in configs/phase4.yaml and documented as the locked "C1" production
# floor, but every run_step3() call site simply omitted the kwargs -- there was no
# error, no warning, just a silent fall-through to the function's permissive defaults
# (mask_all_nodes=False) for an extended period while the docs and config claimed
# otherwise. This is a class of bug that can happen again to any config-controlled
# run_step3 parameter (a future edit that drops a kwarg during a refactor, a copy-
# pasted call site that misses one line). assert_run_step3_wired() closes that hole:
# call it with the SAME cfg dict and the SAME kwargs dict you are about to hand to
# run_step3(**kwargs) at every call site, and it raises immediately if any
# config-controlled parameter is missing from kwargs or doesn't match cfg.

# Maps run_step3() keyword-argument name -> the configs/phase4.yaml key that must
# drive it. Keep in sync with run_step3's signature; every entry here is expected to
# be explicitly present (and equal) in every production/eval call's kwargs dict.
STEP3_PARAM_TO_CFG_KEY: dict[str, str] = {
    "blur_sigma": "blur_sigma",
    "close_disk_r": "morph_close_disk",
    "open_disk_r": "morph_open_disk",
    "min_branch_len_px": "min_branch_len_px",
    "endpoint_bind_extra_px": "endpoint_bind_extra_px",
    "off_page_border_px": "off_page_border_px",
    "mask_all_nodes": "mask_all_nodes",
    "mask_all_nodes_fallback": "mask_all_nodes_fallback",
    "short_gap_max_px": "short_gap_max_px",
    "interpose_tol_px": "short_gap_interpose_tol_px",
    "corridor_half_width_px": "short_gap_corridor_half_width_px",
    "continuity_frac": "short_gap_continuity_frac",
    "perp_reject_ratio": "short_gap_perp_reject_ratio",
    "stub_angle_tol_deg": "short_gap_stub_angle_tol_deg",
}


def assert_run_step3_wired(cfg: dict, call_kwargs: dict) -> None:
    """Fail loudly if a run_step3() call site would silently ignore configs/phase4.yaml.

    Call this with the exact kwargs dict you are about to pass to
    `run_step3(erased_img, nodes, img_w, img_h, **call_kwargs)`, e.g.:

        step3_kwargs = dict(
            blur_sigma=cfg["blur_sigma"], close_disk_r=cfg["morph_close_disk"], ...,
            stub_angle_tol_deg=cfg["short_gap_stub_angle_tol_deg"],
        )
        assert_run_step3_wired(cfg, step3_kwargs)
        s3 = run_step3(erased_bgr, all_symbol_nodes, img_w, img_h, **step3_kwargs)

    Raises AssertionError listing every offending parameter if:
      - a config-controlled kwarg is missing from call_kwargs entirely (the exact
        failure mode that let mask_all_nodes run unwired for an extended period), or
      - a config-controlled kwarg's value doesn't match configs/phase4.yaml (e.g. a
        stale hardcoded literal left over from before a key existed in the yaml).

    Keys in STEP3_PARAM_TO_CFG_KEY absent from `cfg` are skipped (lets older/partial
    config dicts pass without requiring every key to exist everywhere).
    """
    missing: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str, object, object]] = []
    for param_name, cfg_key in STEP3_PARAM_TO_CFG_KEY.items():
        if cfg_key not in cfg:
            continue
        if param_name not in call_kwargs:
            missing.append((param_name, cfg_key))
            continue
        if call_kwargs[param_name] != cfg[cfg_key]:
            mismatched.append((param_name, cfg_key, call_kwargs[param_name], cfg[cfg_key]))

    if not missing and not mismatched:
        return

    lines = ["run_step3() config-wiring assertion failed -- a documented "
             "configs/phase4.yaml setting would be silently ignored:"]
    for param_name, cfg_key in missing:
        lines.append(
            f"  MISSING: call_kwargs has no '{param_name}' -- "
            f"configs/phase4.yaml['{cfg_key}'] is not wired into this run_step3() call "
            f"and its hardcoded default will be used instead."
        )
    for param_name, cfg_key, got, want in mismatched:
        lines.append(
            f"  MISMATCH: call_kwargs['{param_name}']={got!r} but "
            f"configs/phase4.yaml['{cfg_key}']={want!r}"
        )
    raise AssertionError("\n".join(lines))
