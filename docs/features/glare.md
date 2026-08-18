# Glare

**Module:** `app/quality/classical/glare.py`
**Type:** Classical (no model, deterministic)
**Requires:** color image (BGR/BGRA) — grayscale has no saturation channel to check

## What it measures

Specular highlights / glare: pixels that are both bright (high V) AND
washed-out/colorless (low S) in HSV space — the signature of a reflection
or light source blowing out detail, as opposed to a genuinely bright but
colored surface.

## How it works

```python
calculate_glare_mask(image, saturation_threshold=60, value_threshold=230) -> mask
calculate_glare_area_ratio(...) -> float          # total glare pixel fraction
count_glare_regions(..., min_region_area=25) -> int  # distinct glare blobs
calculate_largest_glare_region_ratio(...) -> float   # single largest blob's area fraction
```

1. Convert to HSV
2. Flag pixels where `saturation < 60 AND value > 230`
3. `count_glare_regions` filters out small isolated bright pixels (sensor
   noise, specular dust) below `min_region_area`, so only meaningfully
   sized glare blobs count
4. `largest_glare_region_ratio` isolates the single biggest contiguous
   glare blob — a stronger quality signal than many small scattered
   bright pixels

## API output

```json
"glare": {
  "glare_area_ratio": 0.113,
  "glare_region_count": 4,
  "largest_glare_region_ratio": 0.065
}
```

`glare` is `null` in the API response when the input is grayscale (no
saturation channel to evaluate) — same pattern as occlusion/framing on
grayscale input.

## Suitability-model training: severity injection

`scripts/synthetic_augmentation.py`'s `apply_glare()` draws a blurred white
circular patch at a random position, sized as a fraction of image area:

| Severity | Area fraction | Blend alpha |
|---|---|---|
| low | 3% | 0.5 |
| medium | 8% | 0.75 |
| high | 18% | 0.95 |

## Known limitations

- HSV thresholds (60/230) are fixed defaults, not calibrated against
  labeled glare/no-glare examples.
- A genuinely white/colorless subject (white shirt, snow, paper) can
  false-positive as glare — the detector cannot distinguish "washed out by
  reflection" from "genuinely white."