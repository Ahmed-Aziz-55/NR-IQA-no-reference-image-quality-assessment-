"""
Runs every currently-implemented detector against a single image and
collects the raw results, then feeds the ORIGINAL 7-feature set into
the trained Suitability model for a final verdict.

Framing is now a real detector (ModelFramingDetector) rather than
NotImplementedMetrics, but it is deliberately NOT fed into
_assess_suitability yet — the Suitability model was trained on the
7 features it had when scripts/generate_training_dataset.py ran, and
folding framing in would require regenerating the training set and
retraining, which is a separate task (no decision-engine redesign
as part of wiring framing into the API).
"""

import numpy as np
import pandas as pd

from app.api.schemas import (
    AssessmentResponse,
    BlurMetrics,
    ExposureMetrics,
    FramingMetrics,
    GlareMetrics,
    MotionMetrics,
    OcclusionMetrics,
    ResolutionMetrics,
    SuitabilityMetrics,
)
from app.quality.classical.blur import calculate_laplacian_variance
from app.quality.classical.exposure import (
    calculate_bright_pixel_ratio,
    calculate_dark_pixel_ratio,
    calculate_mean_brightness,
    calculate_saturated_pixel_ratio,
)
from app.quality.classical.glare import (
    calculate_glare_area_ratio,
    calculate_largest_glare_region_ratio,
    count_glare_regions,
)
from app.quality.classical.motion import (
    assess_motion_blur,
    calculate_directional_gradient_variance,
)
from app.quality.classical.resolution import (
    calculate_aspect_ratio,
    calculate_total_pixels,
    get_image_dimensions,
    is_below_minimum_resolution,
)
from app.quality.semantic.framing import FramingResult, ModelFramingDetector
from app.quality.semantic.occlusion import ModelOcclusionDetector, OcclusionResult

# Placeholder default — NOT calibrated. Override per-request via the
# /assess?min_width=&min_height= query params to match your actual
# downstream model's real input requirement. See docs/research.md
# "Low Resolution" for why a fixed default shouldn't be trusted blindly.
DEFAULT_MIN_WIDTH = 640
DEFAULT_MIN_HEIGHT = 480

# Must match scripts/train_suitability_model.py's FEATURE_COLUMNS
# exactly — order and names both — since the trained pipeline was fit
# against a DataFrame with these column names. Framing is intentionally
# NOT included — see module docstring.
SUITABILITY_FEATURE_COLUMNS = [
    "blur_laplacian_variance",
    "exposure_mean_brightness",
    "exposure_dark_pixel_ratio",
    "exposure_bright_pixel_ratio",
    "exposure_saturated_pixel_ratio",
    "glare_area_ratio",
    "glare_region_count",
    "glare_largest_region_ratio",
    "motion_gradient_variance_x",
    "motion_gradient_variance_y",
    "motion_blur_ratio",
    "motion_overall_sharpness",
    "motion_is_likely_motion_blur",
    "resolution_width",
    "resolution_height",
    "resolution_total_pixels",
    "resolution_aspect_ratio",
    "resolution_is_below_minimum",
    "occlusion_score",
    "occlusion_detected",
    "occlusion_face_detected",
]


def _assess_blur(image: np.ndarray) -> BlurMetrics:
    return BlurMetrics(laplacian_variance=calculate_laplacian_variance(image))


def _assess_exposure(image: np.ndarray) -> ExposureMetrics:
    return ExposureMetrics(
        mean_brightness=calculate_mean_brightness(image),
        dark_pixel_ratio=calculate_dark_pixel_ratio(image),
        bright_pixel_ratio=calculate_bright_pixel_ratio(image),
        # Tighter threshold than bright_pixel_ratio (250 vs 245) so a
        # well-lit image isn't confused with a clipped/overexposed one —
        # see calculate_saturated_pixel_ratio docstring.
        saturated_pixel_ratio=calculate_saturated_pixel_ratio(image),
    )


def _assess_glare(image: np.ndarray) -> GlareMetrics | None:
    # Glare requires a color image; grayscale input is a valid image
    # for every other detector, so we skip glare rather than fail
    # the whole request.
    if image.ndim != 3:
        return None

    return GlareMetrics(
        glare_area_ratio=calculate_glare_area_ratio(image),
        glare_region_count=count_glare_regions(image),
        largest_glare_region_ratio=calculate_largest_glare_region_ratio(image),
    )


def _assess_motion(image: np.ndarray) -> MotionMetrics:
    # assess_motion_blur combines the directional gradient ratio with
    # overall Laplacian sharpness, so a sharp image with strong
    # one-directional content (sun rays, blinds, text lines) doesn't get
    # misread as motion blur just because its gradient ratio is high.
    assessment = assess_motion_blur(image)
    variance_x, variance_y = calculate_directional_gradient_variance(image)
    return MotionMetrics(
        gradient_variance_x=variance_x,
        gradient_variance_y=variance_y,
        motion_blur_ratio=assessment.ratio,
        motion_blur_direction=assessment.direction,
        overall_sharpness=assessment.sharpness,
        is_likely_motion_blur=assessment.is_likely_motion_blur,
    )


def _assess_resolution(
    image: np.ndarray,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> ResolutionMetrics:
    width, height = get_image_dimensions(image)
    return ResolutionMetrics(
        width=width,
        height=height,
        total_pixels=calculate_total_pixels(image),
        aspect_ratio=calculate_aspect_ratio(image),
        is_below_minimum=is_below_minimum_resolution(
            image, min_width=min_width, min_height=min_height
        ),
        min_width_used=min_width,
        min_height_used=min_height,
    )


def _assess_occlusion(
    image: np.ndarray,
    detector: ModelOcclusionDetector,
) -> OcclusionMetrics:
    # Occlusion requires a color image (face detection needs BGR/BGRA);
    # grayscale input is valid for every other detector, so skip
    # occlusion rather than fail the whole request — mirrors how
    # _assess_glare handles the same grayscale case.
    if image.ndim != 3:
        return OcclusionMetrics(score=1.0, detected=False, face_detected=None)

    result: OcclusionResult = detector.assess(image)
    return OcclusionMetrics(
        score=result.score,
        detected=result.detected,
        face_detected=result.face_detected,
    )


def _assess_framing(
    image: np.ndarray,
    detector: ModelFramingDetector,
) -> FramingMetrics:
    # Framing (U²-NetP) requires a color image, same reasoning as
    # occlusion/glare above.
    if image.ndim != 3:
        return FramingMetrics(score=0.5, detected=False)

    result: FramingResult = detector.assess(image)
    return FramingMetrics(
        score=result.score,
        detected=result.detected,
        touches_edge=result.touches_edge,
        subject_position_score=result.subject_position_score,
        subject_area_ratio=result.subject_area_ratio,
        left_margin=result.left_margin,
        right_margin=result.right_margin,
        top_margin=result.top_margin,
        bottom_margin=result.bottom_margin,
    )


def _assess_suitability(
    blur: BlurMetrics,
    exposure: ExposureMetrics,
    glare: GlareMetrics | None,
    motion: MotionMetrics,
    resolution: ResolutionMetrics,
    occlusion: OcclusionMetrics,
    suitability_model: dict,
) -> SuitabilityMetrics:
    """
    Builds the same feature vector the model was trained on (see
    scripts/feature_extractor.py) from the other detectors' outputs
    already computed above, and returns the model's verdict. Framing
    is NOT included — see module docstring.

    Args:
        suitability_model: The dict produced by joblib.dump in
            train_suitability_model.py — {"pipeline": ..., "feature_columns": ...}.
            Loaded once at API startup, not per-request — see app/api/main.py.
    """
    # Glare is None on grayscale input; feature_extractor.py falls back
    # to zeros in that same situation, so mirror that here.
    glare_area_ratio = glare.glare_area_ratio if glare is not None else 0.0
    glare_region_count = float(glare.glare_region_count) if glare is not None else 0.0
    glare_largest_region_ratio = (
        glare.largest_glare_region_ratio if glare is not None else 0.0
    )

    row = {
        "blur_laplacian_variance": blur.laplacian_variance,
        "exposure_mean_brightness": exposure.mean_brightness,
        "exposure_dark_pixel_ratio": exposure.dark_pixel_ratio,
        "exposure_bright_pixel_ratio": exposure.bright_pixel_ratio,
        "exposure_saturated_pixel_ratio": exposure.saturated_pixel_ratio,
        "glare_area_ratio": glare_area_ratio,
        "glare_region_count": glare_region_count,
        "glare_largest_region_ratio": glare_largest_region_ratio,
        "motion_gradient_variance_x": motion.gradient_variance_x,
        "motion_gradient_variance_y": motion.gradient_variance_y,
        "motion_blur_ratio": motion.motion_blur_ratio,
        "motion_overall_sharpness": motion.overall_sharpness,
        "motion_is_likely_motion_blur": float(motion.is_likely_motion_blur),
        "resolution_width": float(resolution.width),
        "resolution_height": float(resolution.height),
        "resolution_total_pixels": float(resolution.total_pixels),
        "resolution_aspect_ratio": resolution.aspect_ratio,
        "resolution_is_below_minimum": float(resolution.is_below_minimum),
        "occlusion_score": occlusion.score,
        "occlusion_detected": float(occlusion.detected),
        "occlusion_face_detected": float(bool(occlusion.face_detected)),
    }

    pipeline = suitability_model["pipeline"]
    feature_columns = suitability_model["feature_columns"]

    features_df = pd.DataFrame([row], columns=feature_columns)
    prediction = pipeline.predict(features_df)[0]
    probabilities = pipeline.predict_proba(features_df)[0]

    # Class labels are 0/1, where 1 = "suitable" (matches the training
    # CSV's `suitable` column) — look up its index rather than assuming
    # position, in case the underlying estimator ever orders classes
    # differently.
    suitable_index = list(pipeline.classes_).index(1)
    confidence = float(probabilities[suitable_index])

    return SuitabilityMetrics(suitable=bool(prediction), confidence=confidence)


def assess_image(
    image: np.ndarray,
    filename: str,
    occlusion_detector: ModelOcclusionDetector,
    framing_detector: ModelFramingDetector,
    suitability_model: dict,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> AssessmentResponse:
    """
    Run every implemented detector against a single loaded image, then
    the Suitability model on top of the original 7-feature combined
    output (framing is reported but not yet part of that model's
    input — see module docstring).

    Args:
        image: A validated BGR/BGRA/grayscale uint8 image, as returned
            by app.io.image_loader.load_image.
        filename: Original filename, echoed back for traceability.
        occlusion_detector: A pre-loaded ModelOcclusionDetector instance
            (ONNX session + face detector). Loaded once at API startup,
            not per-request — see app/api/main.py.
        framing_detector: A pre-loaded ModelFramingDetector instance
            (U²-NetP ONNX session). Loaded once at API startup, not
            per-request — see app/api/main.py.
        suitability_model: The pre-loaded Suitability model dict (see
            _assess_suitability). Loaded once at API startup, not
            per-request — see app/api/main.py.
        min_width: Minimum acceptable width for the resolution check.
            Set this to match your actual downstream CV model's input
            requirement — do not rely on the default blindly.
        min_height: Minimum acceptable height for the resolution check.
            Same caveat as min_width.

    Returns:
        AssessmentResponse with raw metrics from every implemented
        detector, plus the Suitability model's final verdict.
    """
    blur = _assess_blur(image)
    exposure = _assess_exposure(image)
    glare = _assess_glare(image)
    motion = _assess_motion(image)
    resolution = _assess_resolution(image, min_width=min_width, min_height=min_height)
    occlusion = _assess_occlusion(image, occlusion_detector)
    framing = _assess_framing(image, framing_detector)

    suitability = _assess_suitability(
        blur, exposure, glare, motion, resolution, occlusion, suitability_model
    )

    return AssessmentResponse(
        filename=filename,
        blur=blur,
        exposure=exposure,
        glare=glare,
        motion=motion,
        resolution=resolution,
        occlusion=occlusion,
        framing=framing,
        suitability=suitability,
    )