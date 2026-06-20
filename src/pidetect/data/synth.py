"""Synthetic P&ID generator from the symbol legend. THE data-scarcity unlock.

Phase 0. Crops real symbol instances from the HF dataset to build a glyph
library, then composes synthetic P&ID sheets: random placement, rotation,
scale, connecting lines (solid=process / dashed=signal), ISA-style tags, and
scan-like degradation. Emits YOLO labels and connectivity JSON for Phase 4.

Usage:
    python -m src.pidetect.data.synth
    python -m src.pidetect.data.synth --n 200 --out data/synthetic --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHEET_W: int = 2048
SHEET_H: int = 2048
NC: int = 32
BASE_GLYPH_PX: int = 80   # reference size; scaled by scale_range at placement
MARGIN: int = 120          # minimum distance from sheet edge for symbol centres

_FUNC_CODES: list[str] = [
    "FT", "LT", "PT", "TT", "FIC", "LIC", "PIC", "TIC",
    "AT", "FE", "FC", "LC", "PC", "TC", "ZT", "WT", "QT", "HS",
]

# Elongated/directional glyphs that look wrong at arbitrary angles
_SNAP_CLASSES: frozenset[int] = frozenset({20, 21, 22})

# 1.7d: per-class size overrides
ARROW_IDX: int = 23
VALVE_IDX: frozenset[int] = frozenset(range(14))   # classes 0–13
# Log-uniform in [12, 80] px diagonal → geometric mean ≈ 31 px, covers real ~16 px regime.
ARROW_DIAG_MIN: int = 12
ARROW_DIAG_MAX: int = 80


# ---------------------------------------------------------------------------
# Glyph library
# ---------------------------------------------------------------------------

def build_glyph_library(
    images_dir: Path,
    labels_dir: Path,
    max_per_class: int = 20,
) -> dict[int, list[np.ndarray]]:
    """Crop real symbol instances from the downloaded HF dataset.

    Scans images/train in sorted order; stops early once every class has
    max_per_class crops. The HF set has >=1620 instances per class, so
    no legend-sheet fallback is needed.

    Returns {class_idx: [bgr_crop, ...]} for all 32 classes.
    """
    crops: defaultdict[int, list[np.ndarray]] = defaultdict(list)

    for img_path in sorted((images_dir / "train").glob("*.jpg")):
        if all(len(crops[c]) >= max_per_class for c in range(NC)):
            break  # every class is full — stop loading images

        lbl_path = labels_dir / "train" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            if len(crops[cls]) >= max_per_class:
                continue
            xc, yc, bw, bh = (float(p) for p in parts[1:5])
            x1 = int((xc - bw / 2) * iw)
            y1 = int((yc - bh / 2) * ih)
            x2 = int((xc + bw / 2) * iw)
            y2 = int((yc + bh / 2) * ih)
            # 10% margin on each side so the full symbol is captured
            mx = max(2, int((x2 - x1) * 0.10))
            my = max(2, int((y2 - y1) * 0.10))
            crop = img[max(0, y1 - my): min(ih, y2 + my),
                       max(0, x1 - mx): min(iw, x2 + mx)]
            if crop.size < 100:
                continue
            crops[cls].append(crop.copy())

    return dict(crops)


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _rotate_glyph(glyph: np.ndarray, angle: float) -> np.ndarray:
    """Rotate glyph by angle degrees (expand canvas); white fill on new area."""
    if abs(angle % 360) < 0.5:
        return glyph
    h, w = glyph.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(glyph, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _paste_glyph(
    canvas: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
) -> tuple[int, int, int, int] | None:
    """Composite patch onto canvas centred at (cx, cy).

    Wherever the patch pixel is darker than 200 on any channel, it overwrites
    the canvas (dark ink on white background — no white rectangle halo).
    Returns the placed bounding box (x1, y1, x2, y2) or None if out-of-bounds.
    """
    ph, pw = patch.shape[:2]
    sh, sw = canvas.shape[:2]

    paste_x1, paste_y1 = cx - pw // 2, cy - ph // 2
    paste_x2, paste_y2 = paste_x1 + pw, paste_y1 + ph

    c_x1 = max(0, paste_x1);  c_y1 = max(0, paste_y1)
    c_x2 = min(sw, paste_x2); c_y2 = min(sh, paste_y2)
    if c_x2 <= c_x1 or c_y2 <= c_y1:
        return None

    p_x1 = c_x1 - paste_x1;  p_y1 = c_y1 - paste_y1
    p_x2 = p_x1 + (c_x2 - c_x1)
    p_y2 = p_y1 + (c_y2 - c_y1)

    patch_roi  = patch[p_y1:p_y2, p_x1:p_x2]
    canvas_roi = canvas[c_y1:c_y2, c_x1:c_x2]
    gray = cv2.cvtColor(patch_roi, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    canvas_roi[mask] = patch_roi[mask]
    return (c_x1, c_y1, c_x2, c_y2)


def _has_overlap(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    pad: int = 10,
) -> bool:
    """Return True if box overlaps any box in placed (with padding)."""
    nx1, ny1, nx2, ny2 = box
    nx1 -= pad; ny1 -= pad; nx2 += pad; ny2 += pad
    for px1, py1, px2, py2 in placed:
        if nx1 < px2 and nx2 > px1 and ny1 < py2 and ny2 > py1:
            return True
    return False


def _draw_dashed_line(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """Draw a dashed line from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    drawing = True
    while t < length:
        seg = dash if drawing else gap
        t_end = min(t + seg, length)
        if drawing:
            p1 = (int(x1 + t * dx), int(y1 + t * dy))
            p2 = (int(x1 + t_end * dx), int(y1 + t_end * dy))
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        t = t_end
        drawing = not drawing


def _draw_lines(
    canvas: np.ndarray,
    placed: list[dict],
    rng: np.random.Generator,
    n_connections: int,
) -> list[dict]:
    """Draw process (solid) and signal (dashed) lines between random symbol pairs."""
    connections: list[dict] = []
    if len(placed) < 2:
        return connections
    pairs_seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(connections) < n_connections and attempts < n_connections * 12:
        attempts += 1
        i = int(rng.integers(len(placed)))
        j = int(rng.integers(len(placed)))
        if i == j or (i, j) in pairs_seen or (j, i) in pairs_seen:
            continue
        pairs_seen.add((i, j))
        a, b = placed[i], placed[j]
        pt1 = (a["cx_px"], a["cy_px"])
        pt2 = (b["cx_px"], b["cy_px"])
        color = (35, 35, 35)
        if rng.random() < 0.6:
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
            line_type = "process"
        else:
            _draw_dashed_line(canvas, pt1, pt2, color, thickness=1)
            line_type = "signal"
        connections.append({"from_id": a["id"], "to_id": b["id"],
                             "type": line_type})
    return connections


def _stamp_tags(
    canvas: np.ndarray,
    placed: list[dict],
    rng: np.random.Generator,
    tag_fraction: float = 0.5,
) -> None:
    """Stamp random ISA-style alphanumeric tags near placed symbols."""
    for sym in placed:
        if rng.random() > tag_fraction:
            continue
        code = _FUNC_CODES[int(rng.integers(len(_FUNC_CODES)))]
        num  = int(rng.integers(100, 999))
        tag  = f"{code}-{num}"
        tx = sym["cx_px"] - sym["w_px"] // 2
        ty = max(12, sym["cy_px"] - sym["h_px"] // 2 - 6)
        cv2.putText(canvas, tag, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1,
                    cv2.LINE_AA)


def _add_degradation(
    img: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 5.0,
    blur_prob: float = 0.5,
    jpeg_quality: int = 88,
) -> np.ndarray:
    """Gaussian noise + optional blur + JPEG round-trip for scan-like look."""
    noise = rng.normal(0, noise_std, img.shape)
    img = np.clip(img.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)
    if rng.random() < blur_prob:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _vary_thickness(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly dilate or erode dark lines to simulate different pen/line weights.

    Applied to arrows only (subtask 1.7d). 50% chance of no change; otherwise
    dilate (thicker) or erode (thinner) with a 3- or 5-px kernel.
    """
    if rng.random() < 0.5:
        return patch
    k = int(rng.choice([3, 3, 5]))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark = (gray < 200).astype(np.uint8) * 255
    kernel = np.ones((k, k), np.uint8)
    if rng.random() < 0.5:
        dark = cv2.dilate(dark, kernel, iterations=1)
    else:
        dark = cv2.erode(dark, kernel, iterations=1)
    result = np.full_like(patch, 255)
    result[dark > 0] = 0
    return result


def _add_valve_extension(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    w_px: int,
    h_px: int,
    rng: np.random.Generator,
    stem_prob: float = 0.35,
) -> tuple[int, int, int, int]:
    """Optionally draw a stem / actuator box / flanges extending from a valve glyph.

    Real P&ID valves carry these appendages; the 1.7a diagnosis found predicted boxes
    43% smaller than GT, which suggests the model never learned the full valve footprint.
    Draws the extension directly onto canvas and returns the expanded bbox.

    Returns (x1, y1, x2, y2) — canvas-clipped, suitable for YOLO label writing.
    """
    sh, sw = canvas.shape[:2]
    x1, y1 = cx - w_px // 2, cy - h_px // 2
    x2, y2 = cx + w_px // 2, cy + h_px // 2

    if rng.random() > stem_prob:
        return x1, y1, x2, y2

    color = (35, 35, 35)
    lw = max(1, int(rng.uniform(1.0, 2.5)))
    ext = rng.choice(["stem", "actuator", "flanges"])

    if ext == "stem":
        stem_h = int(rng.uniform(0.3, 0.6) * h_px)
        top = max(0, y1 - stem_h)
        cv2.line(canvas, (cx, y1), (cx, top), color, lw)
        y1 = top

    elif ext == "actuator":
        act_h = int(rng.uniform(0.25, 0.50) * h_px)
        act_w = int(rng.uniform(0.40, 0.80) * w_px)
        ax1 = max(0, cx - act_w // 2)
        ay1 = max(0, y1 - act_h)
        ax2 = min(sw, ax1 + act_w)
        cv2.rectangle(canvas, (ax1, ay1), (ax2, y1), color, lw)
        cv2.line(canvas, (cx, y1), (cx, ay1), color, lw)
        x1, y1 = min(x1, ax1), min(y1, ay1)
        x2 = max(x2, ax2)

    else:  # flanges
        fl = int(rng.uniform(0.15, 0.35) * w_px)
        left = max(0, x1 - fl)
        right = min(sw, x2 + fl)
        cv2.line(canvas, (x1, cy), (left, cy), color, lw)
        cv2.line(canvas, (x2, cy), (right, cy), color, lw)
        x1 = left
        x2 = right

    return max(0, x1), max(0, y1), min(sw, x2), min(sh, y2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_sheet(
    glyphs: dict[int, list[np.ndarray]],
    n_symbols: int = 40,
    sheet_w: int = SHEET_W,
    sheet_h: int = SHEET_H,
    scale_range: tuple[float, float] = (0.5, 1.8),
    rotate: bool = True,
    noise: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], list[dict]]:
    """Compose one synthetic P&ID sheet.

    Parameters
    ----------
    glyphs      : output of build_glyph_library().
    n_symbols   : target number of symbols to place per sheet.
    scale_range : (min, max) multiplier applied to BASE_GLYPH_PX.
    rotate      : if True, use full 360-deg rotation (90-deg snaps for line elements).
    noise       : if True, add background texture and scan degradation.
    rng         : numpy random generator (pass for reproducibility).

    Returns
    -------
    (image_bgr, yolo_boxes, placed_symbols, connections)
      yolo_boxes     : [{"cls", "xc", "yc", "w", "h"}]  (all normalised 0-1)
      placed_symbols : [{"id", "cls", "cx_px", "cy_px", "w_px", "h_px"}]
      connections    : [{"from_id", "to_id", "type"}]
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- background ---
    canvas = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    if noise:
        bg_noise = rng.normal(0, 3, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)

    valid_classes = [c for c in range(NC) if glyphs.get(c)]
    if not valid_classes:
        raise ValueError("Glyph library is empty.")

    placed_bboxes: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for _ in range(n_symbols):
        cls = int(rng.choice(valid_classes))
        pool = glyphs[cls]
        src  = pool[int(rng.integers(len(pool)))]

        # --- scale ---
        if cls == ARROW_IDX:
            # log-uniform in [ARROW_DIAG_MIN, ARROW_DIAG_MAX]: smaller arrows sampled
            # proportionally more often, covering the real ~16 px regime (1.7d fix).
            diag_px = math.exp(
                rng.uniform(math.log(ARROW_DIAG_MIN), math.log(ARROW_DIAG_MAX))
            )
            target = max(4, int(round(diag_px)))
        else:
            s = float(rng.uniform(scale_range[0], scale_range[1]))
            target = int(np.clip(BASE_GLYPH_PX * s, 20, 200))
        h0, w0 = src.shape[:2]
        if w0 == 0 or h0 == 0:
            continue
        if w0 >= h0:
            tw, th = target, max(1, int(h0 * target / w0))
        else:
            th, tw = target, max(1, int(w0 * target / h0))
        resized = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)

        # --- arrow line-weight variation (1.7d) ---
        if cls == ARROW_IDX:
            resized = _vary_thickness(resized, rng)

        # --- rotate ---
        if rotate:
            if cls in _SNAP_CLASSES:
                angle = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
            else:
                angle = float(rng.uniform(0.0, 360.0))
        else:
            angle = 0.0
        patch = _rotate_glyph(resized, angle)
        ph, pw = patch.shape[:2]

        # --- rejection-sample a non-overlapping position ---
        lo_x = MARGIN + pw // 2
        lo_y = MARGIN + ph // 2
        hi_x = sheet_w - MARGIN - (pw - pw // 2)
        hi_y = sheet_h - MARGIN - (ph - ph // 2)
        if lo_x >= hi_x or lo_y >= hi_y:
            continue  # patch is bigger than the available area

        placed_ok = False
        for _ in range(30):
            cx = int(rng.integers(lo_x, hi_x))
            cy = int(rng.integers(lo_y, hi_y))
            tentative = (cx - pw // 2, cy - ph // 2,
                         cx + (pw - pw // 2), cy + (ph - ph // 2))
            if not _has_overlap(tentative, placed_bboxes, pad=8):
                placed_ok = True
                break
        if not placed_ok:
            continue

        # --- composite ---
        paste_bbox = _paste_glyph(canvas, patch, cx, cy)
        if paste_bbox is None:
            continue
        x1, y1, x2, y2 = paste_bbox

        # --- valve footprint extension (1.7d) ---
        # Draws stems/actuators/flanges for ~35% of valves and expands the YOLO bbox
        # to include the full extent — addressing the 43%-too-small prediction issue.
        if cls in VALVE_IDX:
            x1, y1, x2, y2 = _add_valve_extension(
                canvas, (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1, rng
            )

        placed_bboxes.append((x1, y1, x2, y2))
        placed.append({
            "id":    len(placed),
            "cls":   cls,
            "cx_px": (x1 + x2) // 2,
            "cy_px": (y1 + y2) // 2,
            "w_px":  x2 - x1,
            "h_px":  y2 - y1,
        })

    # --- lines (drawn before tags so tags sit on top) ---
    n_conn = max(1, len(placed) // 3)
    connections = _draw_lines(canvas, placed, rng, n_conn)

    # --- tags ---
    _stamp_tags(canvas, placed, rng)

    # --- degradation ---
    if noise:
        canvas = _add_degradation(canvas, rng)

    # --- YOLO labels ---
    yolo_boxes = [
        {
            "cls": sym["cls"],
            "xc":  sym["cx_px"] / sheet_w,
            "yc":  sym["cy_px"] / sheet_h,
            "w":   sym["w_px"]  / sheet_w,
            "h":   sym["h_px"]  / sheet_h,
        }
        for sym in placed
    ]

    return canvas, yolo_boxes, placed, connections


def generate_dataset(
    glyphs: dict[int, list[np.ndarray]],
    n: int,
    out_dir: Path,
    val_fraction: float = 0.2,
    n_symbols: int = 40,
    seed: int = 42,
    **compose_kwargs,
) -> Counter:
    """Generate n synthetic sheets and write to out_dir in YOLO layout.

    Directory structure written:
        out_dir/images/{train,val}/synth_NNNNN.jpg
        out_dir/labels/{train,val}/synth_NNNNN.txt
        out_dir/connectivity/{train,val}/synth_NNNNN.json

    Returns per-class instance Counter across the training split.
    """
    n_train = int(n * (1 - val_fraction))
    splits  = ["train"] * n_train + ["val"] * (n - n_train)

    for split in ("train", "val"):
        for sub in ("images", "labels", "connectivity"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    class_counts: Counter = Counter()

    for i, split in enumerate(splits):
        img, yolo_boxes, placed, connections = compose_sheet(
            glyphs, n_symbols=n_symbols, rng=rng, **compose_kwargs
        )
        stem = f"synth_{i:05d}"

        # image
        cv2.imwrite(
            str(out_dir / "images" / split / f"{stem}.jpg"), img,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        # YOLO label
        label_lines = [
            f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in yolo_boxes
        ]
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(label_lines)
        )

        # connectivity JSON
        conn_data = {
            "sheet":      stem,
            "image_size": [SHEET_W, SHEET_H],
            "symbols":    placed,
            "connections": connections,
        }
        (out_dir / "connectivity" / split / f"{stem}.json").write_text(
            json.dumps(conn_data, indent=2)
        )

        if split == "train":
            for b in yolo_boxes:
                class_counts[b["cls"]] += 1

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i + 1:>4}/{n}]  {split}  {stem}  "
                  f"({len(placed)} symbols placed)")

    return class_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _visualize_samples(
    img_dir: Path,
    lbl_dir: Path,
    out_dir: Path,
    n: int = 5,
) -> None:
    """Draw bounding-box overlays on evenly-spaced sample sheets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs  = sorted(img_dir.glob("*.jpg"))
    step  = max(1, len(imgs) // n)
    for img_path in imgs[::step][:n]:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                xc, yc, bw, bh = (float(p) for p in parts[1:5])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 60, 220), 2)
                cv2.putText(img, str(cls), (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 220), 1,
                            cv2.LINE_AA)
        cv2.imwrite(str(out_dir / img_path.name), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root",   default="data/digitize-pid-yolo/DigitizePID_Dataset",
                    help="source dataset root (images/ and labels/ inside)")
    ap.add_argument("--out",        default="data/synthetic",
                    help="output directory")
    ap.add_argument("--n",          type=int,   default=200,
                    help="number of synthetic sheets to generate")
    ap.add_argument("--n-symbols",  type=int,   default=40,
                    help="target symbols per sheet")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--max-glyphs", type=int,   default=20,
                    help="glyph instances to keep per class in the library")
    ap.add_argument("--samples",    type=int,   default=5,
                    help="overlay images to save to data/synthetic_samples/")
    args = ap.parse_args()

    src = Path(args.src_root)
    out = Path(args.out)
    SEP = "=" * 62

    # --- build glyph library ---
    print(f"\n{SEP}")
    print("Building glyph library ...")
    print(SEP)
    glyphs = build_glyph_library(
        src / "images", src / "labels",
        max_per_class=args.max_glyphs,
    )
    missing = [c for c in range(NC) if not glyphs.get(c)]
    print(f"  classes with glyphs : {len(glyphs)} / {NC}")
    if missing:
        print(f"  WARNING: missing classes: {missing}")
    else:
        print("  all 32 classes represented")
    for cls in range(NC):
        n_g = len(glyphs.get(cls, []))
        print(f"    cls {cls:>2}: {n_g:>3} glyphs")

    # --- generate dataset ---
    print(f"\n{SEP}")
    print(f"Generating {args.n} sheets -> {out}")
    print(f"  n_symbols={args.n_symbols}  seed={args.seed}")
    print(SEP)
    class_counts = generate_dataset(
        glyphs,
        n=args.n,
        out_dir=out,
        n_symbols=args.n_symbols,
        seed=args.seed,
    )

    n_train = int(args.n * 0.8)
    print(f"\n  train sheets : {n_train}")
    print(f"  val sheets   : {args.n - n_train}")
    print(f"  total train boxes : {sum(class_counts.values())}")
    print(f"\n  Per-class train counts:")
    for cls in range(NC):
        print(f"    cls {cls:>2}: {class_counts.get(cls, 0):>5}")
    if missing:
        print(f"\n  WARNING: classes with 0 instances: "
              f"{[c for c in range(NC) if class_counts.get(c, 0) == 0]}")

    # --- visualise samples ---
    samples_dir = Path("data/synthetic_samples")
    _visualize_samples(
        out / "images" / "train",
        out / "labels" / "train",
        samples_dir,
        n=args.samples,
    )
    print(f"\n  {args.samples} overlays -> {samples_dir}/")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
