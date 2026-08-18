# Evaluation

Status of what's been measured/validated vs. what's still a placeholder.

## Occlusion classifier (ConvNeXt-Small, fine-tuned)

| Stage | Metric | Value |
|---|---|---|
| Original checkpoint (upstream, pre-fine-tune) | Val accuracy / F1 | 0.9887 / 0.9884 (on its own mask/sunglasses-style benchmark) |
| Original checkpoint on real hand-over-face images | Confidence | 20.9% and 10.7% occluded-probability on two known-occluded test images — **wrong direction, failure case** |
| Fine-tuned on COFW | Best val_acc | **0.7870** (epoch 14/15) |
| Fine-tuned, regression check | `mask.jpeg` (mask occlusion) | 0.000007 → 0.013 occluded-probability — still confidently correct, no regression |

COFW's test split is deliberately occlusion-heavy (307 occluded / 200
clean) and considered a hard benchmark — 78.7% val_acc on it is a
reasonable, not exceptional, result. Train/val gap (89.2% train_acc vs.
78.7% val_acc) suggests mild overfitting given the small fine-tuning set
(~500 genuinely occluded training images).

**Not yet done:** quantitative re-test on the original two hand-occlusion
failure images (`occl.jpeg`, `zFfvV.jpg`) with the fine-tuned model to
confirm the fix closed the gap it was meant to close — confirmed
qualitatively working via `mask.jpeg` regression check, but the original
failure cases weren't re-verified with numbers in this record.

## Face detection (YuNet vs. Haar cascade)

No formal precision/recall comparison run — validated via targeted test
cases only:

- Haar cascade: failed to detect a face on `mask.jpeg` (0/1 on this case)
- YuNet: detected the same face correctly (`face_detected: true`)

`face_score_threshold = 0.6` is a deliberate choice (vs. YuNet's own 0.9
demo default), not empirically tuned against a labeled set.

## Framing detector

**Unit tests (pure `compute_framing_result()` logic, synthetic masks):**
11/11 passing — centered subject, small subject (above/below noise floor,
tested separately), oversized subject, edge-touching/cropped subject,
empty mask, noise-only mask, cluttered scene (largest-region selection),
custom-threshold override, corner cases.

**Real-image spot check:** landscape photo (no clear single subject) →
correctly returned the neutral "not evaluated" result (`score: 0.5,
detected: false`, all subject-specific fields `null`) rather than a forced
verdict — confirms the empty/near-empty saliency mask path works as
designed on genuinely ambiguous input.

**Not yet done:** no portrait/product-photo test with a clear subject has
been run to confirm `subject_area_ratio`, margins, and `touches_edge`
produce sensible real numbers (only the "no subject found" path has been
exercised on a real image so far).

## Suitability classifier

**First run (7 features, pre-framing), 4000-row dataset** (1435 suitable /
2565 not suitable):

| Model | Accuracy | Precision (suitable) | Recall (suitable) | F1 (suitable) |
|---|---|---|---|---|
| Logistic Regression | 65% | 0.51 | 0.81 | 0.62 |
| Random Forest | 79% | 0.70 | 0.75 | **0.72** |

Random Forest selected (higher F1). Logistic Regression's much lower
accuracy is consistent with the label rule being non-linear (a "high OR 2+
medium" threshold/count rule) — a linear decision boundary can't represent
it well, which is exactly what a tree-based model is suited for.

**Retrained with framing feature included** — dataset regenerated, model
retrained. Current metrics: see `models/suitability_model_metadata.json`
(written by `scripts/train_suitability_model.py` on every training run) —
not duplicated here to avoid this doc going stale; that file is the source
of truth for the currently-deployed model's numbers.

## What's calibrated vs. still a placeholder

Explicitly NOT calibrated against labeled ground truth yet (documented as
such in each source file):

- Occlusion decision threshold (0.5 — the upstream model's own boundary)
- YuNet `face_score_threshold` (0.6)
- All of `FramingThresholds` (edge margin 2%, small/large subject cutoffs
  5%/85%, edge-touch penalty, position weight, detected-score threshold)
- Resolution's default 640×480 minimum (explicitly documented as a
  generic placeholder — the real requirement should come from whatever
  downstream CV model consumes the images)
- The Suitability label rule itself ("high OR 2+ medium") — an engineering
  judgment call, not derived from labeled real-world suitable/unsuitable
  examples

**Recommended before any production threshold-sensitive deployment:**
collect a small labeled validation set (real images, human-judged suitable/
not-suitable, with framing/occlusion ground truth) and recalibrate against
it — every placeholder above is called out in its source file specifically
so this is a search-and-fix task, not an archaeology one.

## Docker

Build + run cycle verified end-to-end: `docker build` completes with no
errors (all layers), `docker run` starts cleanly (all 4 models — occlusion
ONNX, YuNet ONNX, U²-NetP ONNX, Suitability joblib — load successfully at
startup), `/assess` returns `200 OK` with results matching the non-Docker
run on the same input image.

**Not yet done:** no load/concurrency testing, no image-size optimization
pass (the `models/` folder currently includes both the original and
fine-tuned occlusion ONNX — see `docs/decisions.md` cleanup note), no
deployment to the actual target server (pending company VM, per earlier
project notes).