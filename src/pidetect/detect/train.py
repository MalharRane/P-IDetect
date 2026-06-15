"""YOLOv11 training entrypoint (runs on Colab/Kaggle GPU; launched, not edited, there).

Phase 1. Start axis-aligned YOLOv11s/m, then switch to YOLOv11-OBB.
Turn ON rotation augmentation (degrees>0) and scale — the hackathon model didn't.

TODO:
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    model.train(data="configs/yolo_baseline.yaml", imgsz=1024, epochs=100,
                degrees=180, scale=0.5, mosaic=1.0, fliplr=0.5, batch=16)
Pair with tiling (data/tiling.py) so imgsz reflects tiles, not whole sheets.
"""
raise NotImplementedError("Phase 1: implement YOLOv11 training.")
