# Valve B — Root-cause: mislocated detections (subtask 1.7a refinement)

**N = 62 valve GT boxes in bucket B (MISLOCATED, IoU 0.1–0.5)**  
**OPEN100 Tier 2, 12 real sheets, conf threshold 0.01**

For each pair: dx/dy are pred_center − GT_center, normalized by GT box width/height.
area_ratio = pred_area / GT_area. Positive dx = pred shifted right; positive dy = down.

## Center offset and area ratio distributions

| Metric       |   mean |  median |    std |    p25 |    p75 |
|:-------------|-------:|--------:|-------:|-------:|-------:|
| dx           |  0.084 |  0.046 |  0.312 | -0.029 |  0.283 |
| dy           |  0.041 | -0.019 |  0.229 | -0.046 |  0.242 |
| |dx|         |  0.237 |  0.266 |  0.220 |  0.046 |  0.313 |
| |dy|         |  0.171 |  0.108 |  0.158 |  0.034 |  0.260 |
| area_ratio   |  1.391 |  0.567 |  1.398 |  0.487 |  2.258 |

## Directional bias check

- dx bias: +0.084 (positive = pred shifted right)
- dy bias: +0.041 (positive = pred shifted down)
- 90 % of |dx| values ≤ 0.390
- 90 % of |dy| values ≤ 0.411
- 90 % of area_ratio values in [0.426, 4.443]

## Conclusion

Diagnosis: **real localization failure (with secondary box-size underprediction)**

Key observations:
- **Signed bias is near-zero**: dx mean=+0.084, dy mean=+0.041. The model is not
  systematically shifted to one side — the predictions are roughly centered on the
  GT symbol.
- **Center scatter is the primary driver**: |dx| median=0.266 (≈27% of GT width) means
  predictions are scattered around the GT center, not locked to it. This scatter, combined
  with the IoU floor at 0.5, explains most B cases.
- **Predicted boxes are systematically smaller**: area_ratio median=0.567 (pred ≈ 43%
  smaller than GT). However the IQR spans 0.487–2.258, confirming this is *not* a clean
  constant convention offset — OPEN100 GT boxes may be drawn more expansively than our
  synthetic labels in some cases, but the wild variance rules out a uniform annotation gap
  as the sole explanation.

Combined interpretation: the model detects something valve-like near the right location
(near-zero signed bias) but cannot precisely centre on the real-world glyph or bound it
consistently. This is a real localisation difficulty caused by the domain shift — OPEN100
valve symbols look different enough from the synthetic training glyphs that the regression
head does not transfer cleanly. The secondary size underprediction *may* partly reflect a
box-convention difference (synthetic labels tightly cropped; OPEN100 GT includes more
context) but the large variance makes this a secondary signal, not the root cause.
