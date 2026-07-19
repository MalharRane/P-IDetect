"""Phase 4 step 4: junction analysis — classify skeleton intersections.

Two junction types:
  connector — pipes physically join.  Detected as high-degree (>=3 neighbour)
               skeleton pixels; clustered into a single centre per junction.
  crossing  — pipes cross over each other without joining (P&ID bridge-gap
               convention).  Detected as anti-parallel pairs of unbound free-tip
               endpoints separated by a skeleton gap of the right size.

Public API:
    run_step4(skeleton, branches, endpoints, **cfg) -> Step4Result
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pidetect.graph.lines import Branch, BoundEndpoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JunctionNode:
    junc_id: int
    cx: float                           # x pixel centre (col)
    cy: float                           # y pixel centre (row)
    junc_type: str                      # "connector" | "crossing"
    subtype: str                        # connector: "elbow"|"tee"|"4way"|"nway"
                                        # crossing:  "crossing"
    branch_ids: list[int]               # all branch IDs meeting here
    pass_through_pairs: list[tuple[int, int]] = field(default_factory=list)
    # crossing: [(under_A, under_B), (over_X, over_Y)]
    # connector: empty


@dataclass
class Step4Result:
    connectors: list[JunctionNode]
    crossings: list[JunctionNode]

    @property
    def all_junctions(self) -> list[JunctionNode]:
        return self.connectors + self.crossings

    @property
    def n_connectors(self) -> int:
        return len(self.connectors)

    @property
    def n_crossings(self) -> int:
        return len(self.crossings)

    def connector_subtypes(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(c.subtype for c in self.connectors))


# ---------------------------------------------------------------------------
# Skeleton neighbour counting
# ---------------------------------------------------------------------------

_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    return cv2.filter2D(
        skeleton.astype(np.uint8), -1, _KERNEL, borderType=cv2.BORDER_CONSTANT
    )


# ---------------------------------------------------------------------------
# Connector detection (degree->=3 skeleton pixels)
# ---------------------------------------------------------------------------

def _connector_pixel_mask(skeleton: np.ndarray) -> np.ndarray:
    nc = _neighbor_count(skeleton)
    return skeleton & (nc >= 3)


def _cluster_connector_pixels(
    junc_mask: np.ndarray,
    dilation_r: int = 3,
) -> list[tuple[float, float, list[tuple[int, int]]]]:
    """Cluster touching junction pixels; return [(cx, cy, [(row,col),...]), ...]."""
    if not junc_mask.any():
        return []
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_r + 1, 2 * dilation_r + 1)
    )
    dilated = cv2.dilate(junc_mask.astype(np.uint8), disk)
    n_lbl, labels = cv2.connectedComponents(dilated, connectivity=8)

    jrows, jcols = np.where(junc_mask)
    clusters: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(jrows.tolist(), jcols.tolist()):
        lbl = int(labels[r, c])
        clusters.setdefault(lbl, []).append((r, c))

    result: list[tuple[float, float, list[tuple[int, int]]]] = []
    for pixels in clusters.values():
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        result.append((float(np.mean(cols)), float(np.mean(rows)), pixels))
    return result


def _branch_direction_at_tip(branch: Branch, which_end: str, n_px: int = 8) -> float:
    """Angle (degrees, screen coords) pointing INWARD from the tip."""
    coords = branch.coords_rc  # (N,2) row-col
    if len(coords) < 2:
        return 0.0
    if which_end == "src":
        tip = coords[0]
        interior = coords[min(n_px, len(coords) - 1)]
    else:
        tip = coords[-1]
        interior = coords[max(0, len(coords) - 1 - n_px)]
    dy = float(interior[0] - tip[0])  # row diff (y in screen coords, +down)
    dx = float(interior[1] - tip[1])  # col diff (x, +right)
    return math.degrees(math.atan2(dy, dx))


def _branches_at_junction(
    cx: float, cy: float,
    junc_pixels: list[tuple[int, int]],
    branches: list[Branch],
    radius: float,
) -> list[tuple[int, str]]:
    """Return [(branch_id, which_end), ...] for branches touching this junction."""
    junc_set = set(junc_pixels)
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for b in branches:
        if b.branch_id in seen:
            continue
        for which_end, rc in [("src", b.src_rc), ("dst", b.dst_rc)]:
            pr, pc = int(rc[0]), int(rc[1])
            if (pr, pc) in junc_set or math.hypot(pc - cx, pr - cy) <= radius:
                result.append((b.branch_id, which_end))
                seen.add(b.branch_id)
                break
    return result


def _connector_subtype(
    branch_infos: list[tuple[int, str]],
    branches: list[Branch],
    elbow_angle_min: float,
) -> str:
    degree = len(branch_infos)
    if degree < 2:
        return "stub"
    if degree == 2:
        bmap = {b.branch_id: b for b in branches}
        dirs = [
            _branch_direction_at_tip(bmap[bid], wend)
            for bid, wend in branch_infos
            if bid in bmap
        ]
        if len(dirs) == 2:
            diff = abs((dirs[0] - dirs[1]) % 360)
            if diff > 180:
                diff = 360 - diff
            bend = abs(diff - 180)  # 0 = straight, 90 = right angle
            if bend >= elbow_angle_min:
                return "elbow"
        return "straight"
    if degree == 3:
        return "tee"
    if degree == 4:
        return "4way"
    return "nway"


def _pair_gap_score(
    cx: float,
    cy: float,
    dir_out: float,
    binary_raw: np.ndarray,
    probe_len: int = 12,
) -> float:
    """Sample binary_raw along a ray from (cx, cy) in direction dir_out.

    Returns the fraction of non-line (False) pixels in the first probe_len pixels
    out from the junction centre.  High score (close to 1.0) = gap = crossing.
    """
    H, W = binary_raw.shape[:2]
    rad = math.radians(dir_out)
    cos_d, sin_d = math.cos(rad), math.sin(rad)
    false_count = 0
    for k in range(1, probe_len + 1):
        x = int(round(cx + k * cos_d))
        y = int(round(cy + k * sin_d))
        if 0 <= x < W and 0 <= y < H:
            if not binary_raw[y, x]:
                false_count += 1
    return false_count / probe_len


def _reclassify_4way_as_crossing(
    cx: float,
    cy: float,
    branch_infos: list[tuple[int, str]],
    branches: list[Branch],
    binary_raw: np.ndarray,
    collinear_tol: float = 30.0,
    gap_fraction_thresh: float = 0.5,
    probe_len: int = 12,
) -> Optional[JunctionNode]:
    """Try to reclassify a degree-4 junction as a crossing via binary_raw probe.

    At a crossing bridged by morphological closing, binary_raw still has False
    pixels along the underpass direction near the junction centre.  We probe
    binary_raw outward from (cx, cy) along each collinear pair's direction;
    if one pair's outward rays are mostly False (gap fraction > thresh) while
    the other pair's rays are mostly True, the junction is a crossing.

    Returns a JunctionNode(junc_type="crossing") if reclassified, else None.
    """
    bmap = {b.branch_id: b for b in branches}
    if len(branch_infos) != 4:
        return None

    # Outward direction from junction centre along each branch
    dir_infos: list[tuple[float, int, str]] = []
    for bid, wend in branch_infos:
        b = bmap.get(bid)
        if b is None:
            return None
        inward = _branch_direction_at_tip(b, wend)
        outward = (inward + 180.0) % 360.0
        dir_infos.append((outward, bid, wend))

    # Best pairing into two collinear pairs
    best_err = 9999.0
    best_pairing: Optional[tuple[tuple, tuple]] = None
    for a, b, c, d in [(0,1,2,3), (0,2,1,3), (0,3,1,2)]:
        err_ab = _angle_anti_parallel_err(dir_infos[a][0], dir_infos[b][0])
        err_cd = _angle_anti_parallel_err(dir_infos[c][0], dir_infos[d][0])
        total = err_ab + err_cd
        if total < best_err:
            best_err = total
            best_pairing = (
                (dir_infos[a], dir_infos[b]),
                (dir_infos[c], dir_infos[d]),
            )

    if best_pairing is None or best_err > 2 * collinear_tol:
        return None

    # For each pair, probe binary_raw in both outward directions
    pair_gap_scores = []
    for pair in best_pairing:
        scores = [
            _pair_gap_score(cx, cy, info[0], binary_raw, probe_len)
            for info in pair
        ]
        pair_gap_scores.append(sum(scores) / len(scores))

    # Crossing: one pair is mostly gap, the other is mostly line
    g0, g1 = pair_gap_scores
    if g0 < gap_fraction_thresh and g1 < gap_fraction_thresh:
        return None  # both pairs have lines → genuine connector

    if g0 < gap_fraction_thresh and g1 >= gap_fraction_thresh:
        under_pair, over_pair = best_pairing[1], best_pairing[0]
    elif g1 < gap_fraction_thresh and g0 >= gap_fraction_thresh:
        under_pair, over_pair = best_pairing[0], best_pairing[1]
    else:
        # Both gapped (rare) — smaller gap pair = overpass
        if g0 < g1:
            under_pair, over_pair = best_pairing[1], best_pairing[0]
        else:
            under_pair, over_pair = best_pairing[0], best_pairing[1]

    under_ids = (under_pair[0][1], under_pair[1][1])
    over_ids  = (over_pair[0][1],  over_pair[1][1])

    return JunctionNode(
        junc_id=-1,  # set by caller
        cx=cx, cy=cy,
        junc_type="crossing",
        subtype="crossing",
        branch_ids=[bi for bi, _ in branch_infos],
        pass_through_pairs=[under_ids, over_ids],
    )


def detect_connectors(
    skeleton: np.ndarray,
    branches: list[Branch],
    dilation_r: int = 3,
    radius: float = 10.0,
    elbow_angle_min: float = 20.0,
    start_id: int = 0,
    binary_raw: Optional[np.ndarray] = None,
    collinear_tol: float = 30.0,
    min_gap_px: int = 4,
) -> tuple[list[JunctionNode], list[JunctionNode]]:
    """Find and classify skeleton junctions as connectors or crossings.

    Degree-4 junctions are additionally tested against binary_raw (pre-close image)
    to reclassify as crossings when one collinear pair shows a bridge gap.

    Returns (connectors, crossings_from_4way).
    """
    junc_mask = _connector_pixel_mask(skeleton)
    clusters = _cluster_connector_pixels(junc_mask, dilation_r)
    connectors: list[JunctionNode] = []
    crossings_4way: list[JunctionNode] = []
    junc_id = start_id
    for cx, cy, pixels in clusters:
        infos = _branches_at_junction(cx, cy, pixels, branches, radius)
        if len(infos) < 2:
            continue
        # Attempt degree-4 → crossing reclassification first
        if len(infos) == 4 and binary_raw is not None:
            crossing = _reclassify_4way_as_crossing(
                cx, cy, infos, branches, binary_raw, collinear_tol, min_gap_px
            )
            if crossing is not None:
                crossing.junc_id = junc_id
                crossings_4way.append(crossing)
                junc_id += 1
                continue
        subtype = _connector_subtype(infos, branches, elbow_angle_min)
        connectors.append(JunctionNode(
            junc_id=junc_id,
            cx=cx, cy=cy,
            junc_type="connector",
            subtype=subtype,
            branch_ids=[bi for bi, _ in infos],
        ))
        junc_id += 1
    return connectors, crossings_4way


# ---------------------------------------------------------------------------
# Crossing detection (anti-parallel unbound tip pairs with skeleton gap)
# ---------------------------------------------------------------------------

def _angle_anti_parallel_err(a: float, b: float) -> float:
    """How far (degrees) from perfectly anti-parallel (180 apart)."""
    diff = abs((a - b) % 360)
    if diff > 180:
        diff = 360 - diff
    return abs(diff - 180)


def _has_skeleton_gap(
    skeleton: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    min_gap_px: int = 4,
) -> bool:
    """True if >= min_gap_px consecutive non-skeleton pixels exist on line p1->p2."""
    x1, y1 = p1
    x2, y2 = p2
    H, W = skeleton.shape[:2]
    n = max(abs(x2 - x1), abs(y2 - y1)) + 1
    if n < 2:
        return not bool(skeleton[y1, x1])
    xs = np.linspace(x1, x2, n).round().astype(int).clip(0, W - 1)
    ys = np.linspace(y1, y2, n).round().astype(int).clip(0, H - 1)
    run = max_run = 0
    for y, x in zip(ys, xs):
        if not skeleton[y, x]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run >= min_gap_px


def _overpass_branches(
    cx: float, cy: float,
    underpass_ids: set[int],
    branches: list[Branch],
    radius: float,
) -> list[int]:
    """Find branches (not underpass) whose pixels pass near the crossing centre."""
    overpass: list[int] = []
    for b in branches:
        if b.branch_id in underpass_ids:
            continue
        dists = np.hypot(
            b.coords_rc[:, 1].astype(float) - cx,
            b.coords_rc[:, 0].astype(float) - cy,
        )
        if float(dists.min()) <= radius:
            overpass.append(b.branch_id)
    return overpass


def detect_crossings(
    branches: list[Branch],
    endpoints: list[BoundEndpoint],
    skeleton: np.ndarray,
    min_crossing_gap: float = 5.0,
    max_crossing_gap: float = 110.0,
    collinear_tol: float = 30.0,
    min_gap_px: int = 4,
    search_radius: float = 15.0,
    start_id: int = 0,
    binary_raw: Optional[np.ndarray] = None,
    exclude_rects: Optional[list[tuple[float, float, float, float]]] = None,
) -> list[JunctionNode]:
    """Detect crossings from pairs of opposing unbound free-tip endpoints.

    For each pair (A, B) of unbound, non-off-page, type-0/1 branch tips:
      1. Distance in [min_crossing_gap, max_crossing_gap]
      2. Inward directions anti-parallel within collinear_tol
      3. Skeleton gap >= min_gap_px between them
    -> CROSSING at midpoint; overpass branches found within search_radius.

    Uses best-match greedy (sorted by gap distance) to avoid wrong pairings.
    binary_raw: pre-close Otsu binary; used for gap check if provided
    (more reliable than the skeleton which may bridge small gaps).
    """
    bmap = {b.branch_id: b for b in branches}
    gap_img = binary_raw if binary_raw is not None else skeleton

    def _in_excluded(px: int, py: int) -> bool:
        if not exclude_rects:
            return False
        return any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in exclude_rects)

    cands = [
        ep for ep in endpoints
        if ep.bound_node_id is None
        and not ep.is_off_page
        and ep.branch_id in bmap
        and bmap[ep.branch_id].branch_type in (0, 1)
        and not _in_excluded(*ep.point_xy)
    ]

    dirs: list[float] = [
        _branch_direction_at_tip(bmap[ep.branch_id], ep.which_end)
        for ep in cands
    ]

    # Collect ALL valid pairs first, then match greedily by gap distance (closest = most confident)
    all_pairs: list[tuple[float, float, int, int]] = []  # (gap_dist, angle_err, i, j)
    for i in range(len(cands)):
        ax, ay = cands[i].point_xy
        for j in range(i + 1, len(cands)):
            if cands[j].branch_id == cands[i].branch_id:
                continue
            bx, by = cands[j].point_xy
            gap_dist = math.hypot(ax - bx, ay - by)
            if not (min_crossing_gap <= gap_dist <= max_crossing_gap):
                continue
            angle_err = _angle_anti_parallel_err(dirs[i], dirs[j])
            if angle_err > collinear_tol:
                continue
            if not _has_skeleton_gap(gap_img, (ax, ay), (bx, by), min_gap_px):
                continue
            all_pairs.append((gap_dist, angle_err, i, j))

    # Sort by gap_dist (smallest first — tightest crossing gap = most reliable)
    all_pairs.sort(key=lambda t: (t[0], t[1]))

    crossings: list[JunctionNode] = []
    used: set[int] = set()
    junc_id = start_id

    for gap_dist, angle_err, i, j in all_pairs:
        ep_A = cands[i]; ep_B = cands[j]
        if ep_A.branch_id in used or ep_B.branch_id in used:
            continue

        ax, ay = ep_A.point_xy
        bx, by = ep_B.point_xy
        cx = (ax + bx) / 2.0
        cy = (ay + by) / 2.0
        under_ids = {ep_A.branch_id, ep_B.branch_id}
        over_ids = _overpass_branches(cx, cy, under_ids, branches, search_radius)

        # Require at least one overpass branch through the crossing area.
        # Two stubs facing across an erased symbol have no branch between them.
        if not over_ids:
            continue

        pass_through = [(ep_A.branch_id, ep_B.branch_id)]
        if len(over_ids) >= 2:
            pass_through.append((over_ids[0], over_ids[1]))
        elif len(over_ids) == 1:
            pass_through.append((over_ids[0], over_ids[0]))

        crossings.append(JunctionNode(
            junc_id=junc_id,
            cx=cx, cy=cy,
            junc_type="crossing",
            subtype="crossing",
            branch_ids=sorted(under_ids) + over_ids,
            pass_through_pairs=pass_through,
        ))
        junc_id += 1
        used.update(under_ids)

    return crossings


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_step4(
    skeleton: np.ndarray,
    branches: list[Branch],
    endpoints: list[BoundEndpoint],
    collinear_angle_tol: float = 30.0,
    min_crossing_gap: float = 5.0,
    max_crossing_gap: float = 110.0,
    elbow_angle_min: float = 20.0,
    junction_radius_R: int = 10,
    min_gap_px: int = 4,
    binary_raw: Optional[np.ndarray] = None,
    exclude_rects: Optional[list[tuple[float, float, float, float]]] = None,
) -> Step4Result:
    """Classify all skeleton junctions as connectors or crossings.

    Two crossing-detection paths:
    1. Unbound free-tip pairs (type-0/1 branches) facing each other with gap.
    2. Degree-4 skeleton junctions whose collinear pair has a bridge gap in binary_raw.

    binary_raw: pre-closing Otsu binary from Step3Result; enables path 2.
    """
    connectors, crossings_4way = detect_connectors(
        skeleton, branches,
        dilation_r=3,
        radius=float(junction_radius_R),
        elbow_angle_min=elbow_angle_min,
        start_id=0,
        binary_raw=binary_raw,
        collinear_tol=collinear_angle_tol,
        min_gap_px=min_gap_px,
    )
    crossings_tip = detect_crossings(
        branches, endpoints, skeleton,
        min_crossing_gap=min_crossing_gap,
        max_crossing_gap=max_crossing_gap,
        collinear_tol=collinear_angle_tol,
        min_gap_px=min_gap_px,
        search_radius=float(junction_radius_R) + 5.0,
        start_id=len(connectors) + len(crossings_4way),
        binary_raw=binary_raw,
        exclude_rects=exclude_rects,
    )
    all_crossings = crossings_4way + crossings_tip
    return Step4Result(connectors=connectors, crossings=all_crossings)
