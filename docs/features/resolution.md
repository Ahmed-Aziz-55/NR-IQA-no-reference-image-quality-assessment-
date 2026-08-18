# Low Resolution

**Module:** `app/quality/classical/resolution.py`
**Type:** Classical (no model, deterministic)

## What it measures

Whether an image's pixel dimensions meet a minimum requirement, plus basic
dimension metrics.

```python
get_image_dimensions(image) -> (width, height)
calculate_total_pixels(image) -> int
calculate_aspect_ratio(image) -> float          # width / height
is_below_minimum_resolution(image, min_width=640, min_height=480) -> bool
```

## Why the 640×480 default is explicitly NOT a real requirement

**This is the single most emphasized caveat in this codebase.** "Low
resolution" is only meaningful relative to what a specific downstream CV
model actually needs as input. A model that works fine at 320×320 should
not reject a 612×546 image just because it's below an arbitrary 640×480
cutoff.

`min_width`/`min_height` are always meant to be passed explicitly by the
caller (the API's `/assess?min_width=&min_height=` query params, not
currently exposed — see `main.py`'s `_DEFAULT_MIN_WIDTH`/
`_DEFAULT_MIN_HEIGHT`) — never relied on as-is.

## API output

```json
"resolution": {
  "width": 388,
  "height": 515,
  "total_pixels": 199820,
  "aspect_ratio": 0.753,
  "is_below_minimum": true,
  "min_width_used": 640,
  "min_height_used": 480
}
```

`min_width_used`/`min_height_used` are echoed back specifically so callers
can see exactly what threshold was applied, rather than silently assuming
the default.

## A bug this caveat predicted, and it happened

KonIQ-10k (used as the Suitability model's generic-image source) is a
**fixed** 512×384 — already below the 640×480 default on every single
image. Early in Suitability-model development, `apply_low_resolution()`
downscaled then upscaled back to original dimensions (to fake a "low-res
look"), so the actual pixel shape never changed — `is_below_minimum` was
constant (`true`) across the entire dataset regardless of injected
severity, giving this feature zero training variance.

**Fix:** `scripts/synthetic_augmentation.py`'s `apply_resolution()` now
does a REAL resize to different target dimensions (not a blur-based fake),
straddling the 640×480 default so both true/false values actually occur in
training data:

| Severity | Target dimensions |
|---|---|
| none | 1024×768 |
| low | 800×600 |
| medium | 480×360 |
| high | 240×180 |

(All 4:3, matching KonIQ's native aspect ratio.)

## Known limitations

- Resolution alone doesn't capture *effective* resolution — an upscaled
  low-quality image can report a high pixel count while still looking soft
  (that's `blur.py`'s job, not this module's).