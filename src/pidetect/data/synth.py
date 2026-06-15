"""Synthetic P&ID generator from the symbol legend. THE data-scarcity unlock.

Phase 0. Paste legend glyphs onto backgrounds at random rotation/scale/position,
draw process (solid) + signal (dashed) lines between them, stamp random tags.
Emits images + YOLO labels (and later, line/connection labels for Phase 4).

TODO:
- load_glyphs(legend_dir) -> {class_name: [glyph images]}
- compose_sheet(glyphs, n_symbols, rotate=True, scale_range=(...), noise=True)
- emit YOLO labels; optionally emit graph ground-truth for Phase 4
This is also the headline LinkedIn devlog: "infinite labeled P&IDs from a legend."
"""
raise NotImplementedError("Phase 0: implement the synthetic generator.")
