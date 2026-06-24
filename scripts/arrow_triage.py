"""Arrow triage — Phase 1.8 decision session.

Generates:
  docs/arrow_triage/gt_vs_synth.png       — 30 GT OPEN100 arrows vs 10 synth idx-23
  docs/arrow_triage/examples/             — 3 new annotated context crops
  docs/arrow_triage/miss_breakdown.md     — bucket counts + 8 example refs + verdict

No model weights required. Runs on local machine.

Usage:
    PYTHONPATH=src python scripts/arrow_triage.py [options]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_box_pil(draw: ImageDraw.ImageDraw,
                   box: tuple[float, float, float, float],
                   color: str = "lime", width: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _thumb(crop: Image.Image, size: int) -> Image.Image:
    """Resize crop to fit within sizexsize, centred on white canvas."""
    scale = min(size / max(crop.width, 1), size / max(crop.height, 1))
    new_w = max(1, int(crop.width * scale))
    new_h = max(1, int(crop.height * scale))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _crop_with_box(img: Image.Image,
                    x1: float, y1: float, x2: float, y2: float,
                    pad: int,
                    box_color: str = "lime",
                    box_width: int = 2) -> Image.Image:
    """Crop a padded region from img, draw a box at (x1,y1,x2,y2) relative to crop."""
    W, H = img.size
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(W, int(x2) + pad)
    cy2 = min(H, int(y2) + pad)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    # box in crop-local coordinates
    bx1 = int(x1) - cx1
    by1 = int(y1) - cy1
    bx2 = int(x2) - cx1
    by2 = int(y2) - cy1
    draw = ImageDraw.Draw(crop)
    _draw_box_pil(draw, (bx1, by1, bx2, by2), color=box_color, width=box_width)
    return crop


# ---------------------------------------------------------------------------
# Load OPEN100 GT arrows from graphml
# ---------------------------------------------------------------------------

def load_open100_arrows(raw_dir: Path) -> list[dict]:
    """Return all 'arrow' nodes across 12 sheets as dicts with pixel coords + size."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pidetect.data.open100 import parse_graphml

    arrows = []
    for i in range(12):
        gml = raw_dir / f"{i}.graphml"
        png = raw_dir / f"{i}.png"
        if not (gml.exists() and png.exists()):
            print(f"  [warn] missing sheet {i}, skipping")
            continue
        with Image.open(png) as im:
            img_w, img_h = im.size
        for label, x1, y1, x2, y2 in parse_graphml(gml):
            if label != "arrow":
                continue
            w = x2 - x1
            h = y2 - y1
            diag = math.sqrt(w * w + h * h)
            near_edge = (x1 < 5 or y1 < 5
                         or x2 > img_w - 5 or y2 > img_h - 5)
            arrows.append({
                "sheet": i,
                "png": png,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h, "diag": diag,
                "img_w": img_w, "img_h": img_h,
                "near_edge": near_edge,
            })
    return arrows


# ---------------------------------------------------------------------------
# Load synthetic flow_arrow (idx=23) from training data
# ---------------------------------------------------------------------------

def load_synth_arrows(images_dir: Path, labels_dir: Path,
                       n: int, seed: int) -> list[tuple[Path, float, float, float, float]]:
    """Sample n synthetic idx-23 instances as (img_path, xc, yc, bw, bh) normalised."""
    from pidetect.data.inspect import collect_class_instances
    by_class = collect_class_instances(images_dir, labels_dir)
    instances = by_class.get(23, [])  # idx 23 = flow_arrow
    rng = random.Random(seed)
    chosen: list[tuple[Path, tuple]] = []
    per_img: dict[Path, int] = defaultdict(int)
    pool = instances[:]
    rng.shuffle(pool)
    for img_path, box in pool:
        if per_img[img_path] >= 2:
            continue
        chosen.append((img_path, box))
        per_img[img_path] += 1
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for img_path, box in pool:
            if (img_path, box) not in chosen:
                chosen.append((img_path, box))
                if len(chosen) >= n:
                    break
    return [(p, *box) for p, box in chosen[:n]]


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(
    gt_arrows: list[dict],
    synth_instances: list[tuple],
    out_path: Path,
    n_gt: int = 30,
    n_synth: int = 10,
    thumb: int = 120,
    pad: int = 40,
    cols_gt: int = 6,
    seed: int = 42,
) -> None:
    """Build gt_vs_synth.png: 30 GT crops (top) + 10 synth crops (bottom)."""
    rng = random.Random(seed)
    # sample n_gt from GT arrows across diverse sheets
    by_sheet: dict[int, list[dict]] = defaultdict(list)
    for a in gt_arrows:
        by_sheet[a["sheet"]].append(a)
    sampled_gt: list[dict] = []
    pool_gt = gt_arrows[:]
    rng.shuffle(pool_gt)
    per_sheet: dict[int, int] = defaultdict(int)
    max_per_sheet = max(1, n_gt // max(len(by_sheet), 1) + 2)
    for a in pool_gt:
        if per_sheet[a["sheet"]] >= max_per_sheet:
            continue
        sampled_gt.append(a)
        per_sheet[a["sheet"]] += 1
        if len(sampled_gt) >= n_gt:
            break
    # top-up if needed
    for a in pool_gt:
        if a not in sampled_gt:
            sampled_gt.append(a)
            if len(sampled_gt) >= n_gt:
                break
    sampled_gt = sampled_gt[:n_gt]

    # --- crop GT thumbnails ---
    gt_thumbs: list[Image.Image] = []
    for a in sampled_gt:
        with Image.open(a["png"]) as im:
            crop = _crop_with_box(im, a["x1"], a["y1"], a["x2"], a["y2"],
                                   pad=pad, box_color="lime", box_width=2)
        gt_thumbs.append(_thumb(crop, thumb))

    # --- crop synth thumbnails ---
    synth_thumbs: list[Image.Image] = []
    for img_path, xc, yc, bw, bh in synth_instances:
        with Image.open(img_path) as im:
            W, H = im.size
            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H
            crop = _crop_with_box(im, x1, y1, x2, y2,
                                   pad=pad, box_color="lime", box_width=2)
        synth_thumbs.append(_thumb(crop, thumb))

    # --- layout ---
    font_hdr = _try_font(18)
    font_small = _try_font(14)
    hdr_h = 32
    sep_h = 44

    cols_synth = n_synth
    rows_gt = math.ceil(len(gt_thumbs) / cols_gt)
    rows_synth = 1

    total_w = max(cols_gt, cols_synth) * thumb
    total_h = (hdr_h + rows_gt * thumb + sep_h + rows_synth * thumb + 8)

    sheet = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(sheet)

    # GT header
    draw.text((8, 6), f"OPEN100 GT 'arrow' instances  (n={len(gt_thumbs)}, "
              f"from 12 real NUCLEAR P&ID sheets)  —  green box = GT annotation",
              fill="black", font=font_hdr)

    # paste GT thumbs
    y_off = hdr_h
    for idx, thumb_img in enumerate(gt_thumbs):
        row, col = divmod(idx, cols_gt)
        sheet.paste(thumb_img, (col * thumb, y_off + row * thumb))
    # light grid lines
    for c in range(1, cols_gt):
        draw.line([(c * thumb, y_off), (c * thumb, y_off + rows_gt * thumb)],
                  fill=(220, 220, 220), width=1)
    for r in range(rows_gt + 1):
        draw.line([(0, y_off + r * thumb), (cols_gt * thumb, y_off + r * thumb)],
                  fill=(220, 220, 220), width=1)

    # separator
    sep_y = hdr_h + rows_gt * thumb
    draw.line([(0, sep_y + 8), (total_w, sep_y + 8)], fill=(160, 160, 160), width=2)
    draw.text((8, sep_y + 14), "Synthetic training  flow_arrow  (idx=23, "
              f"n={len(synth_thumbs)})  —  scale in original ~5000px sheet is ~79px median diagonal",
              fill="#555555", font=font_small)

    # paste synth thumbs
    synth_y = sep_y + sep_h
    for idx, thumb_img in enumerate(synth_thumbs):
        sheet.paste(thumb_img, (idx * thumb, synth_y))
    for c in range(1, cols_synth):
        draw.line([(c * thumb, synth_y), (c * thumb, synth_y + thumb)],
                  fill=(220, 220, 220), width=1)
    draw.line([(0, synth_y), (cols_synth * thumb, synth_y)],
              fill=(220, 220, 220), width=1)
    draw.line([(0, synth_y + thumb), (cols_synth * thumb, synth_y + thumb)],
              fill=(220, 220, 220), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path))
    print(f"  Saved contact sheet -> {out_path}")


# ---------------------------------------------------------------------------
# Size statistics + bucket heuristics
# ---------------------------------------------------------------------------

def size_stats(arrows: list[dict]) -> dict:
    diags = [a["diag"] for a in arrows]
    diags_sorted = sorted(diags)
    n = len(diags_sorted)

    def pct(threshold: float) -> int:
        return sum(1 for d in diags if d < threshold)

    tiny   = pct(8)
    small  = sum(1 for d in diags if 8 <= d < 30)
    medium = sum(1 for d in diags if 30 <= d < 60)
    large  = sum(1 for d in diags if d >= 60)
    near_edge = sum(1 for a in arrows if a["near_edge"])

    def percentile(p: float) -> float:
        idx = int(round(p / 100 * (n - 1)))
        return diags_sorted[max(0, min(idx, n - 1))]

    return {
        "n": n,
        "mean": sum(diags) / n if n else 0,
        "median": percentile(50),
        "p25": percentile(25),
        "p75": percentile(75),
        "min": diags_sorted[0] if n else 0,
        "max": diags_sorted[-1] if n else 0,
        "tiny": tiny,
        "small": small,
        "medium": medium,
        "large": large,
        "near_edge": near_edge,
    }


def triage_bucket(w: float, h: float, near_edge: bool) -> str:
    """Heuristic bucket for one arrow.  (d) requires visual inspection, not assigned here."""
    diag = math.sqrt(w * w + h * h)
    if diag < 8 or (diag < 15 and near_edge):
        return "b"   # too small/truncated for reliable GT
    if diag < 30:
        return "c"   # scale gap — training synth arrows are 5x larger
    return "a"       # large enough the model should have detected


# ---------------------------------------------------------------------------
# Annotated context examples
# ---------------------------------------------------------------------------

def make_example(
    arrow: dict,
    label: str,
    out_path: Path,
    context_px: int = 300,
) -> None:
    """Crop a context region centred on the arrow, draw GT box + bucket label."""
    cx = (arrow["x1"] + arrow["x2"]) / 2
    cy = (arrow["y1"] + arrow["y2"]) / 2
    half = context_px // 2
    W, H = arrow["img_w"], arrow["img_h"]
    rx1 = max(0, int(cx - half))
    ry1 = max(0, int(cy - half))
    rx2 = min(W, rx1 + context_px)
    ry2 = min(H, ry1 + context_px)

    with Image.open(arrow["png"]) as im:
        crop = im.crop((rx1, ry1, rx2, ry2)).convert("RGB")

    bx1 = arrow["x1"] - rx1
    by1 = arrow["y1"] - ry1
    bx2 = arrow["x2"] - rx1
    by2 = arrow["y2"] - ry1

    draw = ImageDraw.Draw(crop)
    _draw_box_pil(draw, (bx1, by1, bx2, by2), color="lime", width=2)

    # label strip at bottom
    font = _try_font(14)
    txt = (f"Sheet {arrow['sheet']}  |  diag={arrow['diag']:.1f}px  "
           f"w={arrow['w']:.0f}  h={arrow['h']:.0f}  near_edge={arrow['near_edge']}  "
           f"|  {label}")
    strip_h = 24
    strip = Image.new("RGB", (crop.width, strip_h), (40, 40, 40))
    ImageDraw.Draw(strip).text((4, 4), txt, fill="white", font=font)

    combined = Image.new("RGB", (crop.width, crop.height + strip_h), "white")
    combined.paste(crop, (0, 0))
    combined.paste(strip, (0, crop.height))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(out_path))
    print(f"  Saved example -> {out_path}  [{label}]")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    arrows: list[dict],
    stats: dict,
    bucket_counts: dict[str, int],
    n_missed_est: int,
    out_path: Path,
    examples_dir: Path,
) -> None:
    """Write docs/arrow_triage/miss_breakdown.md."""
    n = stats["n"]
    reused_dir = out_path.parent.parent / "ood_examples"

    def pct(v: int, total: int) -> str:
        return f"{v} ({100 * v / max(total, 1):.1f}%)"

    lines = [
        "# Arrow Miss Triage — Bucket Breakdown",
        "",
        f"**Date:** 2026-06-24  ",
        f"**GT source:** OPEN100 Tier-2, 12 real nuclear P&ID sheets (PID2Graph, CC BY-SA 4.0)  ",
        f"**Eval metric:** CtrMt@50% = 65.2% (phase 1.8c) -> ~34.8% missed  ",
        f"**Raw GT arrows:** {n} unique (451 from graphml); 833 after tiling (overlap creates duplicates)  ",
        "",
        "---",
        "",
        "## Question 1 — Is OPEN100's 'arrow' the same class as our flow_arrow?",
        "",
        "**GraphML schema findings:**",
        "- OPEN100 uses a single `arrow` label — no sub-types in the 10-label vocabulary.",
        f"- Total across 12 sheets: {n} arrow nodes.",
        "- Typical bbox: 9–15 px diagonal (full-sheet pixel coords at ~4000–5000 px sheet size).",
        "",
        "**Structural answer:** Cannot confirm class identity from labels alone.",
        "OPEN100's annotation scheme covers ALL directional arrowheads on a nuclear reactor P&ID —",
        "which likely includes:",
        "  1. Flow direction indicators (solid filled triangles on process/utility lines) = our `flow_arrow`",
        "  2. Signal/instrument connection arrowheads (on dashed lines between instrument bubbles",
        "     and control valves) = NOT in our training distribution",
        "  3. Possibly off-page connector arrows = NOT our class",
        "",
        "**Visual inspection required:** See `docs/arrow_triage/gt_vs_synth.png`.",
        "Compare top section (30 GT OPEN100 'arrow' crops) with bottom section (10 synth `flow_arrow`).",
        "",
        "**Verdict placeholder — fill in after visual inspection:**",
        "> OPEN100's 'arrow' is [SAME / BROADER] than our `flow_arrow`.",
        "> Estimated [N]% of GT arrows appear to be non-flow arrowheads (bucket d).",
        "",
        "---",
        "",
        "## Question 2 — Why are the ~35% missed?",
        "",
        "### Size distribution — ALL raw OPEN100 arrows (n={})".format(n),
        "",
        "| Band | Count | % | Notes |",
        "|---|---|---|---|",
        f"| Tiny  (<8px diag)   | {stats['tiny']}  | {100*stats['tiny']/n:.1f}% | Sub-pixel / likely unreliable GT |",
        f"| Small (8–30px diag) | {stats['small']} | {100*stats['small']/n:.1f}% | Scale gap — synth median is 79.2px |",
        f"| Medium (30–60px)    | {stats['medium']}| {100*stats['medium']/n:.1f}% | Borderline detectable |",
        f"| Large (>60px)       | {stats['large']} | {100*stats['large']/n:.1f}% | Should be detectable |",
        f"| Near image edge     | {stats['near_edge']} | {100*stats['near_edge']/n:.1f}% | Possibly truncated |",
        "",
        f"Summary stats: mean={stats['mean']:.1f}px  median={stats['median']:.1f}px  "
        f"p25={stats['p25']:.1f}px  p75={stats['p75']:.1f}px  "
        f"range=[{stats['min']:.1f}, {stats['max']:.1f}]",
        "",
        f"Synthetic training arrows (idx=23, n=2579): median=79.2px diagonal  **-> scale ratio 0.{int(stats['median']/79.2*1000):03d}x**",
        "",
        "### Bucket assignment (heuristic, for the ~{} estimated missed unique arrows)".format(n_missed_est),
        "",
        "Bucket rule applied to ALL {n} raw arrows; the ~{m} missed arrows are "
        "approximately the smallest ones (consistent with A-bucket median 15.9px in "
        "`docs/ood_arrow_rootcause.md`).".format(n=n, m=n_missed_est),
        "",
        "| Bucket | Definition | Count (all) | Est. % of missed |",
        "|---|---|---|---|",
        f"| (a) Genuine model miss   | diag ≥ 30px (detectable range)  "
        f"| {bucket_counts['a']} | ~{100*bucket_counts['a']/max(n_missed_est,1):.0f}% (upper bound) |",
        f"| (b) GT quality issue     | diag < 8px OR (diag < 15px AND near edge) "
        f"| {bucket_counts['b']} | ~{100*bucket_counts['b']/max(n_missed_est,1):.0f}% (upper bound) |",
        f"| (c) Training dist. gap   | 8 ≤ diag < 30px, scale 5x smaller than synth "
        f"| {bucket_counts['c']} | ~{100*bucket_counts['c']/max(n_missed_est,1):.0f}% |",
        f"| (d) Semantic mismatch    | not a flow_arrow (signal/instrument arrows) "
        f"| ? | VISUAL REVIEW — see gt_vs_synth.png |",
        "",
        "**Note:** (d) cannot be assigned by size alone. "
        "Fill in after inspecting `docs/arrow_triage/gt_vs_synth.png`.",
        "",
        "### Comparison with 1.7a OOD bucket analysis (IoU-based)",
        "",
        "| Old bucket | Count | % of GT | Maps to new bucket |",
        "|---|---|---|---|",
        "| TP (IoU ≥ 0.5)   | 316 | 37.9% | (matched) |",
        "| A NO_FIRE         | 291 | 34.9% | (c) + (b) + (d) |",
        "| B MISLOCATED      | 219 | 26.3% | (c) + (a) |",
        "| C WRONG_CLASS     |   7 |  0.8% | (a) |",
        "| D EXCLUDED_IDX    |   0 |  0.0% | — |",
        "",
        "CtrMt@50% (65.2%) is more lenient than IoU@0.5 — it counts MISLOCATED as hit",
        "if the prediction center lands within 50% of GT box. So the 34.8% CtrMt miss",
        "≈ the hardest subset of the A+B combined (47.2%) where no prediction came close.",
        "",
        "---",
        "",
        "## 8 Annotated Examples",
        "",
        "### Examples 1–2: Bucket A (NO_FIRE — model fired nothing nearby)",
        "",
        f"![arrow_A_001](../ood_examples/arrow_bucket_A_001.png)  ",
        f"![arrow_A_002](../ood_examples/arrow_bucket_A_002.png)  ",
        "",
        "### Examples 3–4: Bucket B (MISLOCATED — prediction existed but IoU < 0.5)",
        "",
        f"![arrow_B_001](../ood_examples/arrow_bucket_B_001.png)  ",
        f"![arrow_B_002](../ood_examples/arrow_bucket_B_002.png)  ",
        "",
        "### Example 5: Bucket C (WRONG_CLASS — prediction well-placed but wrong supercategory)",
        "",
        f"![arrow_C_001](../ood_examples/arrow_bucket_C_001.png)  ",
        "",
        "### Examples 6–7: Potential bucket (d) candidates — visual review required",
        "",
        f"![d_candidate_01](examples/d_candidate_01.png)  ",
        f"![d_candidate_02](examples/d_candidate_02.png)  ",
        "",
        "### Example 8: Scale context — tiny flow arrow on process line",
        "",
        f"![scale_context_01](examples/scale_context_01.png)  ",
        "",
        "---",
        "",
        "## Verdict",
        "",
        "**One-line ceiling attribution:**",
        "> [FILL IN AFTER VISUAL INSPECTION: Training-distribution problem / Eval-definition problem / Model problem]",
        "",
        "**Preliminary (pre-inspection) verdict:**",
        "> The ceiling is primarily a **training-distribution problem** (c dominates).",
        f"> Synthetic arrows are 5x larger than OPEN100 real arrows (synth median 79.2px vs real median {stats['median']:.1f}px).",
        "> Resolution fixes (1.8a/b/c) proved this hypothesis wrong — scale retrain didn't move the number.",
        "> The actual fix needs to change the training data, not the training resolution.",
        "> If visual inspection shows (d) > 20%, reclassify as partially eval-definition problem",
        "> and re-scope the metric to exclude non-flow arrowheads.",
        "",
        "**One-line recommendation:**",
        "> [FILL IN: accept-and-document / defer / fix-now]",
        "",
        "**Preliminary recommendation:**",
        "> **accept-and-document.** The cheap fix (harvest real arrow crops as synth templates)",
        "> requires a held-out OPEN100 slice and a data-augmentation pass — not a full retrain.",
        "> Do not gate Phase 2 on arrows. Phase 2 (fine-grained classifier) starts regardless.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved report -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Arrow triage — Phase 1.8")
    p.add_argument("--raw-dir", default="data/realworld_eval/open100/_raw",
                   help="Directory with 12 .png + .graphml sheets")
    p.add_argument("--synth-img",
                   default="data/digitize-pid-yolo/DigitizePID_Dataset/images/train")
    p.add_argument("--synth-lbl",
                   default="data/digitize-pid-yolo/DigitizePID_Dataset/labels/train")
    p.add_argument("--out-dir", default="docs/arrow_triage",
                   help="Output directory for all triage outputs")
    p.add_argument("--n-gt", type=int, default=30)
    p.add_argument("--n-synth", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    repo = Path(__file__).parent.parent
    raw_dir   = repo / args.raw_dir
    synth_img = repo / args.synth_img
    synth_lbl = repo / args.synth_lbl
    out_dir   = repo / args.out_dir

    print("=== Arrow Triage ===")

    # 1. Load GT arrows
    print("\n[1] Loading OPEN100 GT arrows …")
    arrows = load_open100_arrows(raw_dir)
    print(f"    Loaded {len(arrows)} arrow annotations across 12 sheets")

    # 2. Load synth arrows
    print(f"\n[2] Loading {args.n_synth} synthetic flow_arrow (idx=23) instances …")
    synth = load_synth_arrows(synth_img, synth_lbl, args.n_synth, args.seed)
    print(f"    Loaded {len(synth)} instances")

    # 3. Contact sheet
    print(f"\n[3] Building contact sheet (n_gt={args.n_gt}, n_synth={args.n_synth}) …")
    make_contact_sheet(
        arrows, synth,
        out_path=out_dir / "gt_vs_synth.png",
        n_gt=args.n_gt,
        n_synth=args.n_synth,
        seed=args.seed,
    )

    # 4. Size statistics
    print("\n[4] Size statistics …")
    stats = size_stats(arrows)
    print(f"    n={stats['n']}  mean={stats['mean']:.1f}px  median={stats['median']:.1f}px  "
          f"p25={stats['p25']:.1f}  p75={stats['p75']:.1f}  "
          f"min={stats['min']:.1f}  max={stats['max']:.1f}")
    print(f"    Tiny (<8px):     {stats['tiny']}  ({100*stats['tiny']/stats['n']:.1f}%)")
    print(f"    Small (8-30px):  {stats['small']}  ({100*stats['small']/stats['n']:.1f}%)")
    print(f"    Medium (30-60px):{stats['medium']}  ({100*stats['medium']/stats['n']:.1f}%)")
    print(f"    Large (>60px):   {stats['large']}  ({100*stats['large']/stats['n']:.1f}%)")
    print(f"    Near edge:       {stats['near_edge']}  ({100*stats['near_edge']/stats['n']:.1f}%)")

    # 5. Bucket heuristics
    print("\n[5] Bucket heuristics …")
    bucket_counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}
    for a in arrows:
        b = triage_bucket(a["w"], a["h"], a["near_edge"])
        bucket_counts[b] += 1
    n_missed_est = int(round(len(arrows) * 0.348))
    print(f"    Est. missed unique arrows: {n_missed_est} (34.8% of {len(arrows)})")
    for bucket, count in bucket_counts.items():
        print(f"    ({bucket}) {count} ({100*count/len(arrows):.1f}% of all; "
              f"est. {100*count/max(n_missed_est,1):.0f}% of missed — upper bound)")

    # 6. Generate 3 example images
    print("\n[6] Generating example images …")
    ex_dir = out_dir / "examples"

    # Sort by diagonal to find useful candidates for each bucket type
    sorted_arrows = sorted(arrows, key=lambda a: a["diag"])

    # scale_context_01: smallest flow-arrow-like arrow (tiny, likely bucket c/b)
    candidates_small = [a for a in sorted_arrows if 8 <= a["diag"] < 20]
    if candidates_small:
        rng = random.Random(args.seed)
        scale_ex = rng.choice(candidates_small[:20]) if len(candidates_small) >= 20 else candidates_small[len(candidates_small)//2]
        make_example(scale_ex,
                     label="Bucket (c) candidate — scale gap: tiny arrow in process context",
                     out_path=ex_dir / "scale_context_01.png",
                     context_px=400)

    # d_candidate_01: medium-small arrow (could be signal-line arrowhead)
    candidates_medium = [a for a in sorted_arrows if 15 <= a["diag"] < 40]
    if candidates_medium:
        rng2 = random.Random(args.seed + 1)
        d_ex1 = rng2.choice(candidates_medium[:15]) if len(candidates_medium) >= 15 else candidates_medium[0]
        make_example(d_ex1,
                     label="VISUAL REVIEW: bucket (d) candidate — is this a signal arrowhead or flow arrow?",
                     out_path=ex_dir / "d_candidate_01.png",
                     context_px=400)

    # d_candidate_02: different sheet for variety
    candidates_d2 = [a for a in candidates_medium if a["sheet"] != d_ex1.get("sheet", -1)] \
        if candidates_medium else []
    if candidates_d2:
        rng3 = random.Random(args.seed + 2)
        d_ex2 = rng3.choice(candidates_d2[:15]) if len(candidates_d2) >= 15 else candidates_d2[0]
        make_example(d_ex2,
                     label="VISUAL REVIEW: bucket (d) candidate — is this a signal arrowhead or flow arrow?",
                     out_path=ex_dir / "d_candidate_02.png",
                     context_px=400)
    elif candidates_medium and len(candidates_medium) > 1:
        make_example(candidates_medium[1],
                     label="VISUAL REVIEW: bucket (d) candidate — is this a signal arrowhead or flow arrow?",
                     out_path=ex_dir / "d_candidate_02.png",
                     context_px=400)

    # 7. Write report
    print("\n[7] Writing miss_breakdown.md …")
    write_report(
        arrows=arrows,
        stats=stats,
        bucket_counts=bucket_counts,
        n_missed_est=n_missed_est,
        out_path=out_dir / "miss_breakdown.md",
        examples_dir=ex_dir,
    )

    print("\n=== Done ===")
    print(f"  Contact sheet: {out_dir / 'gt_vs_synth.png'}")
    print(f"  Examples:      {ex_dir}/")
    print(f"  Report:        {out_dir / 'miss_breakdown.md'}")
    print()
    print("NEXT STEP: Open docs/arrow_triage/gt_vs_synth.png and visually inspect:")
    print("  - Do GT rows show filled-triangle flow arrows (same as synth)?")
    print("  - Or do you see open/thin arrowheads on dashed lines (signal arrows)?")
    print("  Then fill in the [FILL IN] placeholders in miss_breakdown.md.")


if __name__ == "__main__":
    main()
