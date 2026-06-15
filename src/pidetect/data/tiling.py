"""SAHI-style tiling: slice large P&IDs into overlapping tiles for train & inference.

Phase 0/1. Full sheets are ~5000-7000px; a 640px model sees nothing without this.
Build with the `sahi` package (slice_coco / get_sliced_prediction) or a custom slicer.

TODO:
- slice_dataset(images_dir, labels_dir, tile=1024, overlap=0.2) -> tiled YOLO dataset
- merge_predictions(tile_preds) -> full-image boxes (NMS across tile seams)
"""
raise NotImplementedError("Phase 0/1: implement tiling with SAHI.")
