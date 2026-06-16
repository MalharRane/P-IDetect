"""Subtask 1.6c -- build the "open100" real-world eval tier.

Downloads the 12 real, public PID2Graph OPEN100 sheets (lazily, via HTTP
range reads -- see src/pidetect/data/open100.py for why), maps their coarse
graphml labels onto our verified class identities as far as they overlap,
tiles everything at 640px/20% overlap, and writes:

    data/realworld_eval/open100/images/test/*.jpg
    data/realworld_eval/open100/labels/test/*.txt   (3-class: valve/arrow/instrument)
    data/realworld_eval/open100/ignore/test/*.txt    (suppression-only, not scored)
    data/realworld_eval/open100/open100.yaml
    data/realworld_eval/open100/SOURCES.md

Usage (from project root):
    python scripts/build_open100_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pidetect.data.open100 import download_open100, build_open100_tier

RAW_DIR = Path("data/realworld_eval/open100/_raw")
OUT_DIR = Path("data/realworld_eval/open100")

YAML_TEXT = """\
# Built by scripts/build_open100_eval.py -- subtask 1.6c.
# 3-class supercategory space (NOT our 32-class taxonomy) -- see SOURCES.md
# for the mapping from OPEN100's graphml labels onto these, and
# docs/realworld_eval_protocol.md for how evaluate.py scores against it.
path: data/realworld_eval/open100
train: images/test   # required by Ultralytics even for test-only sets; reuse test
val:   images/test
test:  images/test

nc: 3
names:
  0: valve
  1: arrow
  2: instrument
"""

SOURCES_TEXT = """\
# OPEN100 tier -- sources & attribution

**Sheets:** 12 real P&ID sheets from Energy Impact Center's OPEN100 open
nuclear reactor design (https://www.open-100.com/). OPEN100 materials are
freely redistributable (FSF-style public license; export-control-exempted
per DOE/NNSA).

**Graph annotations:** PID2Graph dataset, Sturmer, J. M., Graumann, M., &
Koch, T. (2025). "From Engineering Diagrams to Graphs: Digitizing P&IDs with
Transformers." 2025 IEEE 12th DSAA. Zenodo record 14803338, CC BY-SA 4.0.
https://doi.org/10.5281/zenodo.14803338

**Download method:** the Zenodo record bundles all subsets into one 9.3GB
zip; we only need the 12 "Complete/PID2Graph OPEN100/{0..11}.{png,graphml}"
entries (~11MB), so `src/pidetect/data/open100.py:download_open100` reads the
remote zip lazily via HTTP range requests instead of pulling the full archive.

## Label mapping (the honest overlap)

OPEN100's graphml node labels are far coarser than our 32 verified classes
(see docs/class_identity/mapping.md) -- only 10 distinct values total. Mapped
as follows (single source of truth: `src/pidetect/data/open100.py`):

| OPEN100 label | -> | our supercategory | our indices it stands in for |
|---|---|---|---|
| `valve` | -> | 0 (valve) | 0,1,2,3,4,5,6,7,8,9,10,11,12,13 (bowtie family + the 4 confidently-named valve classes -- excludes 14/15/24 [ambiguous] and 16/17/18 [verified NOT a valve, see docs/phase1_analysis.md]) |
| `arrow` | -> | 1 (arrow) | 23 (`flow_arrow`) |
| `instrumentation` | -> | 2 (instrument) | 25,26,27,28,30 (circle-shaped instrument bubbles; excludes 29/31, which are rectangles) |
| `tank`, `pump`, `general`, `inlet/outlet` | -> | *(ignored)* | none -- no equivalent class. Written to `ignore/test/`; predictions overlapping these are excluded from the false-positive count, not penalized as wrong. |
| `connector`, `crossing`, `background` | -> | *(dropped)* | none -- these are graph-topology annotations (junction/crossing markers), not symbol detections, so they're excluded entirely, including from ignore-suppression. |

**Known scoping limitation:** tiles with zero *scored* ground truth are
excluded from the eval image set even if they contain an *ignored* object
(e.g. a tile showing only a tank) -- so false-positive behavior in
pure-ignore regions isn't measured, only suppressed within tiles that already
contain at least one scored (valve/arrow/instrument) object.

**Suppression/recall tradeoff:** a correct prediction that happens to overlap
an ignore box (IoU >= 0.3) is dropped before scoring, same as a wrong one --
so a real detection sitting right next to e.g. a tank can occasionally cost
a tiny bit of recall in exchange for not being counted as a false positive.
Confirmed in testing: a synthetic "perfect" predictor that exactly
reproduces every scored GT box still lands at AP50 0.997 (not 1.000) on the
instrument supercategory for exactly this reason -- expected, not a bug.
"""


def main() -> None:
    print("Step 1: download OPEN100 sheets (lazy range-read, ~11MB)")
    download_open100(RAW_DIR)

    print("\nStep 2: parse + map + tile")
    counts = build_open100_tier(RAW_DIR, OUT_DIR)

    print(f"\n  sheets processed : {counts['sheets']} / 12")
    print(f"  tiles written    : {counts['tiles']}")
    print(f"  scored boxes     : {counts['scored_boxes']}  "
          f"(valve={counts['by_class'][0]}, arrow={counts['by_class'][1]}, "
          f"instrument={counts['by_class'][2]})")
    print(f"  ignore boxes     : {counts['ignore_boxes']}  (tank/pump/general/inlet-outlet)")
    print(f"  dropped boxes    : {counts['dropped_boxes']}  (connector/crossing/background)")

    (OUT_DIR / "open100.yaml").write_text(YAML_TEXT)
    (OUT_DIR / "SOURCES.md").write_text(SOURCES_TEXT)
    print(f"\n  -> {OUT_DIR / 'open100.yaml'}")
    print(f"  -> {OUT_DIR / 'SOURCES.md'}")


if __name__ == "__main__":
    main()
