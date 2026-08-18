# Poor Framing

**Module:** `app/quality/semantic/framing.py`
**Type:** Semantic (model-based, subject-gated)
**Requires:** color image (BGR/BGRA)

## What it measures

Whether the image's subject is well-framed: not cropped by the image
boundary, not too small or too large relative to the frame, reasonably
positioned.

## Architecture

```
U²-NetP saliency inference -> binarize + denoise mask -> largest connected component
    -> geometric measurements (margins, area ratio, center distance)
    -> compute_framing_result() [pure function] -> FramingResult
```

**Deliberately NOT** "U²-Net directly outputs a framing score" — see
`docs/decisions.md` and `docs/research.md` for why that would be wrong
(salient region ≠ always the intended subject).

Two classes implement `FramingDetector`:

- **`HeuristicFramingDetector`** — classical fallback. Canny edge detection
  + largest contour as the candidate subject. Kept, not deleted, same
  policy as occlusion's heuristic.
- **`ModelFramingDetector`** — the primary implementation. See below.

## `ModelFramingDetector`

```python
ModelFramingDetector(
    onnx_model_path,              # models/u2netp.onnx
    thresholds=FramingThresholds(),
)
```

Model: U²-NetP (Qin et al., *Pattern Recognition* 2020), input 320×320 RGB
ImageNet-normalized NCHW, output = per-pixel saliency map (min-max
normalized defensively, since raw output isn't strictly bounded to [0,1]).

## `FramingThresholds` — every tunable in one place

All thresholds are centralized in a single dataclass (not scattered inline)
specifically so recalibration against labeled data later is a search-and-
edit task in one place:

| Threshold | Default | Meaning |
|---|---|---|
| `mask_threshold` | 0.5 | Saliency score cutoff for "this pixel is subject" |
| `min_component_area_ratio` | 0.01 | Components smaller than 1% of image area are noise, not a subject |
| `morph_kernel_size` | 5 | Morphological opening kernel — strips speckle noise before connected-components |
| `edge_margin_threshold` | 0.02 | Subject-bbox edge closer than 2% of the frame to the boundary = "touching edge" (likely cropped) |
| `small_subject_area_ratio` | 0.05 | Below this area fraction, subject is "lost in the frame" |
| `large_subject_area_ratio` | 0.85 | Above this, subject "dominates," likely over-cropped |
| `edge_touch_penalty` | 0.3 | Flat score deduction when `touches_edge` |
| `position_weight` | 0.2 | How much centering contributes to score — deliberately small, see below |
| `detected_score_threshold` | 0.5 | Score below this (plus `touches_edge`) = "framing issue detected" |

**None calibrated against labeled data yet** — documented starting guesses.

## Why centering is a soft signal, not a hard rule

An off-center subject is a compositional choice, not automatically bad
framing. `subject_position_score` contributes only 20% weight
(`position_weight`) to the final score — the dominant signals are subject
size (`size_score`) and edge-touching, both more directly tied to actual
CV-suitability concerns (a cropped or lost subject genuinely hurts
downstream processing; an off-center subject usually doesn't).

## Noise handling

Small disconnected saliency-mask regions (single-pixel noise, tiny
specular artifacts) are explicitly filtered before subject selection:
morphological opening removes speckle, then components below
`min_component_area_ratio` are discarded. An image with no surviving
component (empty/near-empty mask) returns a neutral "not evaluated" result
— confirmed on a real landscape photo (see `docs/evaluation.md`).

## Cluttered scenes

When multiple disconnected salient regions exist (cluttered background,
multiple objects), the LARGEST is selected as the candidate subject —
consistent with the same convention used elsewhere in this project
(`HeuristicOcclusionDetector`, `HeuristicFramingDetector`).

## API output

```json
"framing": {
  "score": 0.5,
  "detected": false,
  "touches_edge": null,
  "subject_position_score": null,
  "subject_area_ratio": null,
  "left_margin": null,
  "right_margin": null,
  "top_margin": null,
  "bottom_margin": null
}
```

All `null` fields together mean "no discernible subject found" — a
distinct state from a low-scoring but subject-present verdict. Real
example (subject found): margins/area-ratio/position score all populated
with actual numbers.

## Testing

11 unit tests against `compute_framing_result()` directly, using synthetic
masks — no ONNX model file required (`tests/unit/quality/test_framing.py`).
Covers: centered subject, small subject (both above and below the noise
floor, tested as two distinct cases), oversized subject, edge-touching/
cropped subject, empty mask, noise-only mask, cluttered scene, custom
threshold overrides, corner cases.

## Known limitations

- Not yet fed into the Suitability model at the time framing was first
  wired into the API — see `docs/dataset.md`/`docs/decisions.md` for the
  staged rollout (later regenerated + retrained to include it).
- No real portrait/product-photo test has confirmed the populated-fields
  path (margins, area ratio) produces sensible numbers on real images —
  only the empty-mask path has an image-based confirmation so far.