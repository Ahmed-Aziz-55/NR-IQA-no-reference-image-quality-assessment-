# Exposure (Darkness / Overexposure)

**Module:** `app/quality/classical/exposure.py`
**Type:** Classical (no model, deterministic)
**Covers project features:** #2 Darkness/Underexposure, #3 Overexposure

## What it measures

Four related brightness metrics on the grayscale image:

| Function | Measures | Default threshold |
|---|---|---|
| `calculate_mean_brightness` | Overall average intensity (0-255) | — |
| `calculate_dark_pixel_ratio` | Fraction of pixels below threshold (underexposed) | 30 |
| `calculate_bright_pixel_ratio` | Fraction of pixels above threshold (bright/light-toned) | 245 |
| `calculate_saturated_pixel_ratio` | Fraction of pixels at/near true sensor clipping | 250 (tighter) |

## Why bright ≠ saturated (two separate metrics)

`bright_pixel_ratio` (>245) is a wide, soft range — a well-lit, light-toned
image (white background, bright daylight) can score high here without
being overexposed. `saturated_pixel_ratio` (≥250) is a tighter cutoff
intended to isolate genuine clipping — detail actually lost because the
sensor hit its maximum value.

A well-lit image can have a high `bright_pixel_ratio` with a near-zero
`saturated_pixel_ratio` — that combination means "bright but not
overexposed," a distinction `mean_brightness` or `bright_pixel_ratio`
alone cannot make.

## API output

```json
"exposure": {
  "mean_brightness": 93.38,
  "dark_pixel_ratio": 0.114,
  "bright_pixel_ratio": 0.00003,
  "saturated_pixel_ratio": 0.00001
}
```

## Suitability-model training: darkness and overexposure are mutually exclusive

In `scripts/synthetic_augmentation.py`, darkness and overexposure severity
are sampled as ONE choice (`sample_exposure_severity`), not independently —
an image is injected as either dark, or bright, or neither, never both at
once, since they're opposite ends of the same physical brightness axis.
Injecting both simultaneously would partially cancel out and produce a
meaningless combined effect.

| Severity | Darkness factor | Overexposure factor |
|---|---|---|
| low | ×0.70 | ×1.3 |
| medium | ×0.45 | ×1.7 |
| high | ×0.20 | ×2.4 |

## Known limitations

- No calibrated "this image is definitively underexposed/overexposed"
  cutoff exists in this module — raw ratios only, thresholding is a
  caller/Suitability-model concern.
- Grayscale-only analysis — doesn't account for color-channel-specific
  clipping (e.g. a blown-out red channel in an otherwise normal-looking
  image).