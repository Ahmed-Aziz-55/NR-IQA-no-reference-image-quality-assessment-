# IQA — Image Quality Assessment for Downstream CV Suitability

Predicts whether an image is suitable for downstream computer-vision processing,
by detecting and scoring 8 quality issues and combining them into a single
trained suitability verdict.

## Features (8/8 implemented)

| # | Feature | Method |
|---|---|---|
| 1 | Blur | Laplacian variance |
| 2 | Darkness / Underexposure | Dark-pixel ratio (exposure detector) |
| 3 | Overexposure | Bright/saturated-pixel ratios (exposure detector) |
| 4 | Glare | HSV saturation/brightness thresholding |
| 5 | Motion artifacts | Directional gradient anisotropy + overall sharpness |
| 6 | Occlusion | ConvNeXt-Small classifier, fine-tuned on COFW, face-gated via YuNet |
| 7 | Poor framing | U²-NetP salient-subject localization + geometric scoring |
| 8 | Low resolution | Width/height vs. configurable minimum |

Plus a trained **Suitability classifier** (Random Forest) that combines all of
the above into one `suitable: true/false` verdict with a confidence score.

See `docs/features/` for one write-up per feature — what it measures, how it's
implemented, its known limitations, and its tunable thresholds.

## Project docs

- [`docs/architecture.md`](docs/architecture.md) — module layout, request flow, why things are split the way they are
- [`docs/dataset.md`](docs/dataset.md) — KonIQ-10k, COFW, and how the Suitability model's training set was built
- [`docs/decisions.md`](docs/decisions.md) — key engineering decisions and why (occlusion model choice, face-gating, framing architecture, label rule, etc.)
- [`docs/evaluation.md`](docs/evaluation.md) — model performance, what's calibrated vs. still a placeholder
- [`docs/research.md`](docs/research.md) — background research behind the occlusion and framing detector choices

## Quickstart

### Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# training/data-prep tools (never needed for the API itself):
# pip install -r requirements-train.txt

python -m uvicorn app.api.main:app --reload --port 8080
```

Required model files under `models/` (not committed to source control —
download/generate separately, see `docs/dataset.md` and `docs/decisions.md`):

- `face_occlusion_finetuned.onnx` (+ `.onnx.data`) — occlusion classifier
- `face_detection_yunet_2023mar.onnx` — face detector (occlusion gating)
- `u2netp.onnx` — salient-object detector (framing)
- `suitability_model.joblib` — trained Suitability classifier

### Docker

```bash
docker build -t iqa-api:latest .
docker run --rm -p 8080:8080 iqa-api:latest
```

Verified working: clean build, clean startup (all 4 models load successfully),
`/assess` returns `200 OK` with results matching the local (non-Docker) run.

## API

### `GET /health`
Liveness check.

### `POST /assess`
Multipart file upload (`file`). Returns raw metrics from every detector plus
the final suitability verdict:

```bash
curl -X POST http://localhost:8080/assess -F "file=@photo.jpg"
```

```json
{
  "filename": "photo.jpg",
  "blur": { "laplacian_variance": 223.2 },
  "exposure": { "mean_brightness": 61.1, "dark_pixel_ratio": 0.22, ... },
  "glare": { "glare_area_ratio": 0.0, ... },
  "motion": { "motion_blur_ratio": 1.4, "is_likely_motion_blur": false, ... },
  "resolution": { "width": 1281, "height": 1920, "is_below_minimum": false, ... },
  "occlusion": { "score": 1.0, "detected": false, "face_detected": false },
  "framing": { "score": 0.5, "detected": false, "touches_edge": null, ... },
  "suitability": { "suitable": true, "confidence": 1.0 }
}
```

Every field except `suitability` is a **raw metric** — no thresholds applied.
`suitability` is the one opinionated field: the trained model's combined verdict.

## Repo layout

```
app/                    production API + detectors (see docs/architecture.md)
scripts/                training/data-prep tools (never shipped in Docker)
tests/unit/quality/     detector unit tests
models/                 ONNX + joblib model files (not committed)
docs/                   this documentation
requirements.txt        runtime dependencies (Docker image)
requirements-train.txt  training-only dependencies (venv only)
Dockerfile
```