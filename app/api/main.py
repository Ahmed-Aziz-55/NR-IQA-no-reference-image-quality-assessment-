import tempfile
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.assess import assess_image
from app.api.schemas import AssessmentResponse
from app.io.image_loader import ImageLoadError, load_image
from app.quality.semantic.framing import ModelFramingDetector
from app.quality.semantic.occlusion import ModelOcclusionDetector

app = FastAPI(
    title="IQA — Image Quality Assessment API",
)

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# Default minimum resolution requirements for downstream CV processing.
# Not exposed to the API user — keeps the upload endpoint simple.
_DEFAULT_MIN_WIDTH = 640
_DEFAULT_MIN_HEIGHT = 480

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Path to the exported face-occlusion ONNX classifier (ConvNeXt-Small,
# fine-tuned on COFW for hand/object occlusion — see
# scripts/fine_tune_occlusion.py).
_OCCLUSION_ONNX_PATH = "models/face_occlusion_finetuned.onnx"

# Path to the YuNet face-detection ONNX model, used to gate the
# occlusion classifier (only run it when a face is actually present).
_FACE_DETECTOR_ONNX_PATH = "models/face_detection_yunet_2023mar.onnx"

# Path to the U²-NetP salient-object-detection ONNX model, used for
# subject localization in the framing detector.
# Download: https://huggingface.co/Heliosoph/u2net-onnx (u2netp.onnx)
_FRAMING_ONNX_PATH = "models/u2netp.onnx"

# Path to the trained Suitability model — see
# scripts/train_suitability_model.py. A dict: {"pipeline": ..., "feature_columns": ...}.
_SUITABILITY_MODEL_PATH = "models/suitability_model.joblib"

# Loaded once at process startup — reused across requests rather than
# reloaded per-request.
_occlusion_detector = ModelOcclusionDetector(
    onnx_model_path=_OCCLUSION_ONNX_PATH,
    face_detector_model_path=_FACE_DETECTOR_ONNX_PATH,
)
_framing_detector = ModelFramingDetector(onnx_model_path=_FRAMING_ONNX_PATH)
_suitability_model = joblib.load(_SUITABILITY_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.post("/assess", response_model=AssessmentResponse)
async def assess_image_route(
    file: UploadFile = File(...),
) -> AssessmentResponse:
    """
    Assess an uploaded image using the IQA detectors.

    """

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{suffix}'. "
                f"Allowed: {sorted(_ALLOWED_SUFFIXES)}"
            ),
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. "
                f"Max size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    # load_image() works with a filesystem path, so save the upload
    # temporarily and remove it after assessment.
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        image = load_image(tmp_path)

        return assess_image(
            image,
            filename=file.filename or tmp_path.name,
            occlusion_detector=_occlusion_detector,
            framing_detector=_framing_detector,
            suitability_model=_suitability_model,
            min_width=_DEFAULT_MIN_WIDTH,
            min_height=_DEFAULT_MIN_HEIGHT,
        )

    except ImageLoadError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        tmp_path.unlink(missing_ok=True)