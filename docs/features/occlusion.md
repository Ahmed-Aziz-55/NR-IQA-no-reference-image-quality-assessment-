# Occlusion

**Module:** `app/quality/semantic/occlusion.py`
**Type:** Semantic (model-based, face-gated)
**Requires:** color image (BGR/BGRA)

## What it measures

Whether a face in the image is occluded (hand, mask, sunglasses, hair,
object covering part of the face).

## Architecture

```
YuNet face detection -> [face found?] -> crop -> ConvNeXt-Small classifier -> OcclusionResult
                              |
                              no -> "not applicable" (score=1.0, detected=false, face_detected=false)
```

Two classes implement `OcclusionDetector`:

- **`HeuristicOcclusionDetector`** — classical, no-training fallback.
  Tiles the image into blocks, flags low-variance ("flat") blocks, finds
  the largest connected flat region. Rationale: occlusion (finger, sticker,
  overlay) tends to create a large flat region vs. real scene content's
  natural texture. Known false positive: genuinely uniform backgrounds
  (sky, studio backdrop, bokeh) also score flat. **Kept as a fallback/test
  double, not deleted**, when `ModelOcclusionDetector` was integrated.

- **`ModelOcclusionDetector`** — the primary implementation. See below.

## `ModelOcclusionDetector`

Model: ConvNeXt-Small, fine-tuned on COFW (see `docs/research.md` for why,
`docs/decisions.md` for the fine-tuning design, `docs/evaluation.md` for
results — best val_acc 0.7870).

```python
ModelOcclusionDetector(
    onnx_model_path,           # models/face_occlusion_finetuned.onnx
    face_detector_model_path,  # models/face_detection_yunet_2023mar.onnx
    occlusion_threshold=0.5,
    face_score_threshold=0.6,  # lower than YuNet's own 0.9 default — see docs/decisions.md
    face_nms_threshold=0.3,
    face_top_k=5000,
)
```

**Why face-gated, not applied to the whole image directly:** the
classifier was trained only on cropped face images — it has no meaning
applied to a generic scene with no face. `face_detected: false` is a
distinct, honest state from `detected: false` (face present, not
occluded).

**Why YuNet, not Haar cascade:** Haar cascade failed to find faces at all
on mask-occluded images (see `docs/decisions.md`). YuNet, trained on WIDER
FACE (includes partial occlusion), fixed this.

Model input: 224×224 RGB, [0,1]-scaled then ImageNet-normalized, NCHW.
Output: 2-class logits, softmax index 1 = occluded probability.

## API output

```json
"occlusion": {
  "score": 0.013,
  "detected": true,
  "face_detected": true
}
```

`score`: 0.0 (fully occluded) to 1.0 (no occlusion). `face_detected: null`
in the API's `FramingMetrics`-equivalent state doesn't apply here —
occlusion's `face_detected` is always `true`/`false`, never `null`, since
the detector always resolves whether a face was found.

## Known limitations

- `occlusion_threshold = 0.5` is the upstream model's own decision
  boundary, not recalibrated for this project's data.
- Face-gating means occlusion is reported as "not applicable" for the
  large majority of generic (non-portrait) downstream-CV images — by
  design, but worth knowing when reading aggregate suitability stats.
- Largest-face selection when multiple faces are present — a reasonable
  stand-in for "the subject" pending a real subject-priority mechanism.