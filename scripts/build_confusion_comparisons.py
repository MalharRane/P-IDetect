"""Subtask 1.6b -- paste the contact sheets of confused class pairs side by
side, so the confusion-matrix claims in docs/phase1_analysis.md can be
checked by eye: are these genuine visual siblings, or a naming coincidence?

Reads docs/class_identity/idx_NN.png (built by build_class_identity_sheets.py)
and writes one composite per confused group to docs/class_identity/confusion_*.png.

Usage (from project root):
    python scripts/build_confusion_comparisons.py
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IDENTITY_DIR = Path("docs/class_identity")

# (output filename, [class indices to place side by side])
GROUPS = [
    ("confusion_idx16_17_18.png", [16, 17, 18]),
    ("confusion_idx03_10.png",    [3, 10]),
]


def main() -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for out_name, indices in GROUPS:
        sheets = [Image.open(IDENTITY_DIR / f"idx_{i:02d}.png") for i in indices]
        gap = 16
        header_h = 30
        w = sum(s.width for s in sheets) + gap * (len(sheets) - 1)
        h = max(s.height for s in sheets) + header_h
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for i, s in enumerate(sheets):
            canvas.paste(s, (x, header_h))
            x += s.width + gap
        draw.text((8, 4), "side-by-side: " + " | ".join(f"idx {i:02d}" for i in indices),
                   fill="black", font=font)
        out = IDENTITY_DIR / out_name
        canvas.save(out)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
