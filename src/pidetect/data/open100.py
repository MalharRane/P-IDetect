"""PID2Graph OPEN100 tier -- real, public P&ID sheets for out-of-distribution
eval (subtask 1.6c). See docs/class_identity/mapping.md for our verified
class identities and data/realworld_eval/open100/SOURCES.md for attribution.

OPEN100 is 12 real CAD P&ID sheets from Energy Impact Center's open nuclear
reactor design (https://www.open-100.com/), redistributed with .graphml graph
annotations by the PID2Graph dataset (Sturmer, Graumann & Koch 2025,
arXiv/DSAA 2025; Zenodo record 14803338, CC BY-SA 4.0). The Zenodo record is
a single 9.3GB zip bundling several unrelated subsets -- we only want the 12
OPEN100 sheets (~11MB), so `download_open100` reads the remote zip lazily via
HTTP range requests instead of pulling the whole archive.

OPEN100's own annotation vocabulary is much coarser than our 32 classes (10
distinct node labels total, vs. our 32) -- see the mapping table in
docs/realworld_eval_protocol.md. `classify_node` is the single source of
truth for that mapping: every OPEN100 label is either "scored" (maps onto one
of three supercategories we're confident about), "ignore" (a real symbol type
we have no equivalent class for -- tank/pump/general/inlet-outlet -- excluded
from scoring but also excluded from false-positive penalty), or "drop"
(connector/crossing/background -- graph-topology annotations, not symbols).

Usage:
    python scripts/build_open100_eval.py
"""
from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import networkx as nx
import requests

from .tiling import slice_image

ZENODO_ZIP_URL = "https://zenodo.org/api/records/14803338/files/PID2Graph.zip/content"
ZIP_PREFIX = "PID2Graph/Complete/PID2Graph OPEN100"
N_SHEETS = 12

# -- the honest overlap: OPEN100's 10-label vocabulary -> our supercategories ----
# "scored": maps onto a supercategory we evaluate. "ignore": a real symbol type
# with no equivalent in our 32 classes -- excluded from GT, but predictions
# landing on it are also excluded from the false-positive count (not "wrong").
# "drop": not a symbol at all (graph-topology annotation) -- excluded entirely.
SCORED_OPEN100_LABELS = {"valve": 0, "arrow": 1, "instrumentation": 2}
IGNORE_OPEN100_LABELS = {"tank", "pump", "general", "inlet/outlet"}
SUPERCATEGORY_NAMES = {0: "valve", 1: "arrow", 2: "instrument"}

# Our verified (subtask 1.6a/1.6b) class indices that fall under each
# OPEN100 supercategory, for remapping model PREDICTIONS at eval time.
# valve: the bowtie-pinch family + the 4 confidently-named valve classes.
#   Excludes idx 14/15/24 (ambiguous per mapping.md) and 16/17/18 (verified in
#   1.6b to be a spectacle-blind-like fitting, not a valve).
OUR_VALVE_IDX = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
OUR_ARROW_IDX = frozenset({23})
# instrument: circle-shaped instrument bubbles only -- excludes idx 29/31
# (rectangles; mapping.md treats those as a different "tag" concept).
OUR_INSTRUMENT_IDX = frozenset({25, 26, 27, 28, 30})


def classify_node(label: str) -> tuple[str, int | None]:
    """Map one OPEN100 graphml node label to ("scored", cls) | ("ignore", None)
    | ("drop", None). `cls` is 0=valve, 1=arrow, 2=instrument when scored."""
    if label in SCORED_OPEN100_LABELS:
        return "scored", SCORED_OPEN100_LABELS[label]
    if label in IGNORE_OPEN100_LABELS:
        return "ignore", None
    return "drop", None


# ---------------------------------------------------------------------------
# Download (lazy HTTP range reads -- avoid the 9.3GB full archive)
# ---------------------------------------------------------------------------

class _HttpRangeFile(io.RawIOBase):
    """Minimal read-only, seekable file-like object backed by HTTP Range
    requests, just enough for stdlib `zipfile` to do random access on a
    remote zip without downloading it."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        r = requests.head(url, allow_redirects=True, timeout=30)
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        end = self.size - 1 if (n is None or n < 0) else min(self.pos + n - 1, self.size - 1)
        if self.pos > end:
            return b""
        resp = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        resp.raise_for_status()
        data = resp.content
        self.pos += len(data)
        return data


def download_open100(dest_dir: Path, force: bool = False) -> None:
    """Pull the 12 OPEN100 Complete-plan sheets (.png + .graphml) from the
    PID2Graph Zenodo record, without downloading the full 9.3GB zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    have_all = all(
        (dest_dir / f"{i}.{ext}").exists()
        for i in range(N_SHEETS) for ext in ("png", "graphml")
    )
    if have_all and not force:
        print(f"  SKIP: {dest_dir} already has all {N_SHEETS} sheets")
        return

    print(f"  Opening remote zip (lazy, range-read) ...")
    f = _HttpRangeFile(ZENODO_ZIP_URL)
    import zipfile
    zf = zipfile.ZipFile(f)
    for i in range(N_SHEETS):
        for ext in ("png", "graphml"):
            out = dest_dir / f"{i}.{ext}"
            if out.exists() and not force:
                continue
            name = f"{ZIP_PREFIX}/{i}.{ext}"
            data = zf.read(name)
            out.write_bytes(data)
            print(f"  -> {out}  ({len(data):,} bytes)")


# ---------------------------------------------------------------------------
# GraphML -> YOLO label lines
# ---------------------------------------------------------------------------

def parse_graphml(path: Path) -> list[tuple[str, float, float, float, float]]:
    """Return every symbol node's (label, xmin, ymin, xmax, ymax) in pixels."""
    g = nx.read_graphml(path)
    out = []
    for _, attrs in g.nodes(data=True):
        if "xmin" not in attrs:
            continue  # edges / nodes without a bounding box
        out.append((
            attrs["label"],
            float(attrs["xmin"]), float(attrs["ymin"]),
            float(attrs["xmax"]), float(attrs["ymax"]),
        ))
    return out


def _to_yolo_lines(boxes: list[tuple[int, float, float, float, float]],
                    img_w: int, img_h: int) -> str:
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the tiled eval tier
# ---------------------------------------------------------------------------

def build_open100_tier(raw_dir: Path, out_dir: Path,
                        tile: int = 640, overlap: float = 0.2) -> dict:
    """Parse all 12 sheets, split nodes into scored/ignore/drop, tile both
    label sets through the same windows as the rest of the pipeline, and
    write data/realworld_eval/open100/{images,labels,ignore}/test/.

    Returns summary counts for the build script to print.
    """
    from PIL import Image

    img_out = out_dir / "images" / "test"
    lbl_out = out_dir / "labels" / "test"
    ign_out = out_dir / "ignore" / "test"
    for d in (img_out, lbl_out, ign_out):
        d.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="pidetect_open100_"))
    ignore_img_tmp = tmp / "ignore_images"
    ignore_img_tmp.mkdir()

    counts = {"sheets": 0, "tiles": 0, "scored_boxes": 0, "ignore_boxes": 0,
              "dropped_boxes": 0, "by_class": {0: 0, 1: 0, 2: 0}}

    for i in range(N_SHEETS):
        png = raw_dir / f"{i}.png"
        gml = raw_dir / f"{i}.graphml"
        if not (png.exists() and gml.exists()):
            print(f"  [warn] missing sheet {i}, skipping")
            continue

        with Image.open(png) as im:
            img_w, img_h = im.size

        scored_boxes: list[tuple[int, float, float, float, float]] = []
        ignore_boxes: list[tuple[int, float, float, float, float]] = []
        for label, x1, y1, x2, y2 in parse_graphml(gml):
            kind, cls = classify_node(label)
            if kind == "scored":
                scored_boxes.append((cls, x1, y1, x2, y2))
                counts["by_class"][cls] += 1
            elif kind == "ignore":
                ignore_boxes.append((0, x1, y1, x2, y2))  # dummy class, never scored
                counts["ignore_boxes"] += 1
            else:
                counts["dropped_boxes"] += 1

        # full-image temp label files, in pixel-derived normalised YOLO format
        scored_lbl = tmp / f"{i}_scored.txt"
        scored_lbl.write_text(_to_yolo_lines(scored_boxes, img_w, img_h))
        ignore_lbl = tmp / f"{i}_ignore.txt"
        ignore_lbl.write_text(_to_yolo_lines(ignore_boxes, img_w, img_h))

        # tile the scored set -- this writes the real image+label tiles
        stats = slice_image(png, scored_lbl, img_out, lbl_out,
                            tile=tile, overlap=overlap, neg_fraction=0.0)
        # tile the ignore set through the SAME deterministic windows (same
        # image, tile, overlap -> identical tile grid) -- only need the
        # label half; point images at a throwaway dir.
        slice_image(png, ignore_lbl, ignore_img_tmp, ign_out,
                   tile=tile, overlap=overlap, neg_fraction=0.0)
        # tiles are named "{stem}_{row}.txt" by source filename stem; the
        # scored pass used `png.stem` ("0", "1", ...) so the ignore pass
        # (same source filename) produces matching stems automatically.

        counts["sheets"] += 1
        counts["tiles"] += stats["tiles_written"]
        counts["scored_boxes"] += len(scored_boxes)

    shutil.rmtree(tmp, ignore_errors=True)
    return counts
