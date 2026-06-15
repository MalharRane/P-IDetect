"""Honest evaluation: per-class AP (NOT just mAP), precision/recall at deploy threshold.

Phase 1+. mAP hides rare-class failure — always print the per-class table.

TODO:
    metrics = model.val(data="configs/yolo_baseline.yaml")
    # print metrics.box.maps (per-class) alongside metrics.box.map50
"""
raise NotImplementedError("Phase 1: implement per-class AP reporting.")
