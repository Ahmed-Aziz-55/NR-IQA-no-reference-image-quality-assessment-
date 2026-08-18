# Motion Artifacts

**Module:** `app/quality/classical/motion.py`
**Type:** Classical (no model, deterministic)

## What it measures

Directional motion blur — smearing along one axis from camera/subject
movement during exposure — distinguished from general softness/focus blur.

## How it works

**Step 1 — directional gradient variance:**
```python
calculate_directional_gradient_variance(image, ksize=3) -> (variance_x, variance_y)
```
Sobel gradients along X and Y. Motion blur suppresses gradient energy along
the axis of motion, so comparing the two variances reveals directional bias.

**Step 2 — motion blur ratio:**
```python
calculate_motion_blur_ratio(image) -> float  # >= 1.0
```
Ratio of the larger to the smaller gradient variance. Near 1.0 = no
dominant blur direction (sharp, or uniform/focus blur). High = strong
directional signature.

**Step 3 — why ratio alone is not enough (the key design decision here):**

A high gradient ratio is a NECESSARY but not SUFFICIENT signal for motion
blur. A perfectly sharp image with strong one-directional content — sun
rays, blinds, text lines, architectural verticals — also produces a high
ratio despite having zero blur. **This was observed directly**: a sharp,
well-lit image was flagged with high confidence by ratio alone, a false
positive.

```python
assess_motion_blur(image, ratio_threshold=3.0, sharpness_threshold=150.0) -> MotionBlurAssessment
```

Only calls an image "likely motion blurred" when **both** hold:
1. Gradient ratio exceeds `ratio_threshold` (directional imbalance exists), AND
2. Overall Laplacian variance (sharpness, reusing `blur.py`'s metric) is
   below `sharpness_threshold` (the image is actually short on
   high-frequency detail overall — consistent with smearing, not sharp
   directional content)

## API output

```json
"motion": {
  "gradient_variance_x": 207.14,
  "gradient_variance_y": 185.16,
  "motion_blur_ratio": 1.12,
  "motion_blur_direction": "vertical",
  "overall_sharpness": 3.63,
  "is_likely_motion_blur": false
}
```

## Suitability-model training: severity injection

`scripts/synthetic_augmentation.py`'s `apply_motion_blur()` convolves with
a directional kernel at a random angle (0-180°):

| Severity | Kernel length |
|---|---|
| low | 7px |
| medium | 15px |
| high | 25px |

## Known limitations

- `ratio_threshold` (3.0) and `sharpness_threshold` (150.0) are documented
  placeholders, not calibrated against labeled data.
- Still vulnerable to the false-positive case in principle if a sharp
  directionally-textured image ALSO happens to have naturally lower
  overall Laplacian variance than the threshold — the combined check
  reduces but doesn't eliminate this failure mode.