# Blur

**Module:** `app/quality/classical/blur.py`
**Type:** Classical (no model, deterministic)

## What it measures

Overall image sharpness via the variance of the Laplacian response. Lower
variance = less high-frequency detail = blurrier image.

## How it works

```python
calculate_laplacian_variance(image, ksize=1) -> float
```

1. Convert to grayscale
2. Apply the Laplacian operator (`cv2.Laplacian`)
3. Return the variance of the result

Higher values → more edge/detail content (sharp). Lower values → smoother,
less-detailed image (blurry). No fixed "blurry" cutoff is applied here —
this is a raw metric; thresholding is a caller/Suitability-model concern.

## API output

```json
"blur": { "laplacian_variance": 3.63 }
```

## Known limitations

- Laplacian variance alone doesn't distinguish blur *type* — a genuinely
  flat/uniform region (clear sky, blank wall) also scores low, same as an
  out-of-focus blur. See `motion.py` for how this project separates
  motion blur from general softness (directional gradient ratio, using
  this same Laplacian variance as one input).
- No fixed severity thresholds are hardcoded in this module — see
  `scripts/synthetic_augmentation.py`'s `_BLUR_KERNELS` for the severity
  levels used in Suitability-model training data (kernel sizes 5/11/21 for
  low/medium/high injected blur).