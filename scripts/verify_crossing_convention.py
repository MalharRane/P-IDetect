"""
scripts/verify_crossing_convention.py

Verification of Phase 4 Decision 1: does OPEN100 use the bridge-gap
convention for crossings, or something else?

Crops a context window around each annotated crossing and connector node,
assembles contact sheets, and reports the visual convention.

Outputs:
  docs/phase4_design/crossings.png         -- sampled crossing node crops
  docs/phase4_design/connectors.png        -- sampled connector node crops
  docs/phase4_design/crossing_convention.md
"""
from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "realworld_eval" / "open100" / "_raw"
OUT_DIR = ROOT / "docs" / "phase4_design"
SHEETS  = [0, 3, 10]

CROP_HALF = 80    # context half-window in px -> 160x160 crops
N_CROSS   = 60    # crossings to show
N_CONN    = 30    # connectors to show
COLS      = 10
THUMB     = 160
PAD       = 6
SEED      = 42


# ---------------------------------------------------------------------------
# Parse topology nodes
# ---------------------------------------------------------------------------

def _topo_nodes(gml: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Return {label -> [(x1, y1, x2, y2), ...]} for crossing + connector."""
    g = nx.read_graphml(gml)
    out: dict[str, list] = {"crossing": [], "connector": []}
    for _, attrs in g.nodes(data=True):
        label = attrs.get("label", "")
        if label not in out:
            continue
        x1 = attrs.get("xmin")
        y1 = attrs.get("ymin")
        x2 = attrs.get("xmax")
        y2 = attrs.get("ymax")
        if x1 is None:
            continue
        out[label].append((int(x1), int(y1), int(x2), int(y2)))
    return out


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _crop(img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Return a (2*half) x (2*half) crop centred on (cx, cy), white-padded."""
    h, w = img.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    patch = img[y1:y2, x1:x2].copy()
    # Pad if near the edge
    pt = max(0, half - cy)
    pb = max(0, cy + half - h)
    pl = max(0, half - cx)
    pr = max(0, cx + half - w)
    if pt or pb or pl or pr:
        patch = np.pad(
            patch, ((pt, pb), (pl, pr), (0, 0)),
            mode="constant", constant_values=255,
        )
    return patch


def _crosshair(arr: np.ndarray, color=(220, 30, 30), arm=12) -> np.ndarray:
    """Draw a red crosshair at the centre of the crop array."""
    arr = arr.copy()
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    arr[cy, max(0, cx - arm):cx + arm] = color
    arr[max(0, cy - arm):cy + arm, cx] = color
    return arr


# ---------------------------------------------------------------------------
# Contact-sheet builder
# ---------------------------------------------------------------------------

def contact_sheet(crops: list[tuple[np.ndarray, str]], out: Path,
                  title: str, cols: int = COLS,
                  thumb: int = THUMB, pad: int = PAD) -> None:
    if not crops:
        print(f"  [warn] no crops for {out.name}")
        return
    rows = (len(crops) + cols - 1) // cols
    lh = 14                            # label height in pixels
    cw = thumb + pad
    ch = thumb + lh + pad
    W  = cols * cw + pad
    H  = rows * ch + pad + 22         # +22 for title bar

    canvas = Image.new("RGB", (W, H), (230, 230, 230))
    draw   = ImageDraw.Draw(canvas)
    draw.text((pad, 4), title, fill=(20, 20, 20))

    for idx, (arr, lbl) in enumerate(crops):
        row, col = divmod(idx, cols)
        x = col * cw + pad
        y = row * ch + pad + 22

        pil = Image.fromarray(arr.astype(np.uint8))
        pil = pil.resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(pil, (x, y))
        draw.text((x + 2, y + thumb + 1), lbl, fill=(40, 40, 40))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    print(f"  Saved {out}  ({len(crops)} crops, {rows} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    all_cross: list[tuple[str, int, int, int, int]] = []  # (sheet, x1,y1,x2,y2)
    all_conn:  list[tuple[str, int, int, int, int]] = []
    imgs: dict[str, np.ndarray] = {}

    for sid in SHEETS:
        gml = RAW_DIR / f"{sid}.graphml"
        png = RAW_DIR / f"{sid}.png"
        if not (gml.exists() and png.exists()):
            print(f"  [warn] sheet {sid} missing, skipping")
            continue
        imgs[str(sid)] = np.array(Image.open(png).convert("RGB"))
        topo = _topo_nodes(gml)
        for box in topo["crossing"]:
            all_cross.append((str(sid), *box))
        for box in topo["connector"]:
            all_conn.append((str(sid), *box))
        print(f"  Sheet {sid}: {len(topo['crossing'])} crossings, "
              f"{len(topo['connector'])} connectors")

    print(f"\n  Total crossings : {len(all_cross)}")
    print(f"  Total connectors: {len(all_conn)}")

    # bbox size stats (should all be ~8x8 integer grid cells)
    def _size_stats(items: list[tuple]) -> None:
        ws = [x2 - x1 for _, x1, y1, x2, y2 in items]
        hs = [y2 - y1 for _, x1, y1, x2, y2 in items]
        print(f"    bbox w: min={min(ws)} max={max(ws)} "
              f"median={sorted(ws)[len(ws)//2]}")
        print(f"    bbox h: min={min(hs)} max={max(hs)} "
              f"median={sorted(hs)[len(hs)//2]}")

    if all_cross:
        print("  Crossing bbox sizes:")
        _size_stats(all_cross)
    if all_conn:
        print("  Connector bbox sizes:")
        _size_stats(all_conn)

    # Sample and crop
    sample_cross = rng.sample(all_cross, min(N_CROSS, len(all_cross)))
    sample_conn  = rng.sample(all_conn,  min(N_CONN,  len(all_conn)))

    def make_crops(
        samples: list[tuple[str, int, int, int, int]],
    ) -> list[tuple[np.ndarray, str]]:
        result = []
        for sid, x1, y1, x2, y2 in samples:
            img = imgs.get(sid)
            if img is None:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            arr = _crop(img, cx, cy, CROP_HALF)
            arr = _crosshair(arr)
            lbl = f"S{sid} ({cx},{cy})"
            result.append((arr, lbl))
        return result

    cross_crops = make_crops(sample_cross)
    conn_crops  = make_crops(sample_conn)

    contact_sheet(
        cross_crops,
        OUT_DIR / "crossings.png",
        f"OPEN100 CROSSING nodes — {len(cross_crops)} sampled "
        f"(sheets {SHEETS}, {2*CROP_HALF}px context, red=annotated centre)",
    )
    contact_sheet(
        conn_crops,
        OUT_DIR / "connectors.png",
        f"OPEN100 CONNECTOR nodes — {len(conn_crops)} sampled "
        f"(sheets {SHEETS}, {2*CROP_HALF}px context, red=annotated centre)",
    )
    print("\nReview the two PNGs to assess the visual convention, then update")
    print("docs/phase4_design/crossing_convention.md with the findings.")


if __name__ == "__main__":
    main()
