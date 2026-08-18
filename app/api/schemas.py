"""
Response schemas for the IQA assessment API.

Kept deliberately flat and metric-first: every field here is a RAW
metric straight from a classical/model detector, not a pass/fail
verdict — EXCEPT `suitability`, which is the one deliberate exception:
it's the trained Suitability model's final verdict, combining every
other metric into a single suitable/not-suitable decision. Everything
else stays raw so callers can inspect the underlying signal themselves.
"""

from pydantic import BaseModel


class BlurMetrics(BaseModel):
    laplacian_variance: float


class ExposureMetrics(BaseModel):
    mean_brightness: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    saturated_pixel_ratio: float


class GlareMetrics(BaseModel):
    glare_area_ratio: float
    glare_region_count: int
    largest_glare_region_ratio: float


class MotionMetrics(BaseModel):
    gradient_variance_x: float
    gradient_variance_y: float
    motion_blur_ratio: float
    motion_blur_direction: str
    overall_sharpness: float
    is_likely_motion_blur: bool


class ResolutionMetrics(BaseModel):
    width: int
    height: int
    total_pixels: int
    aspect_ratio: float
    is_below_minimum: bool
    # Echoed back so callers can see exactly what threshold was applied
    # instead of assuming the API default (640x480). Override per-request
    # via the /assess?min_width=&min_height= query params — see
    # docs/research.md "Low Resolution" for why this must match your
    # actual downstream model's input requirement, not a fixed default.
    min_width_used: int
    min_height_used: int


class OcclusionMetrics(BaseModel):
    """
    Result of ModelOcclusionDetector (ConvNeXt-Small face-occlusion
    classifier, face-gated, fine-tuned on COFW). See
    app/quality/semantic/occlusion.py.
    """

    score: float
    detected: bool
    # True = a face was found and score/detected reflect a real
    # occlusion verdict. False = no face found in the image, so
    # occlusion was not applicable — score/detected are neutral
    # placeholders in that case, not a real verdict.
    face_detected: bool | None = None


class FramingMetrics(BaseModel):
    """
    Result of ModelFramingDetector (U²-NetP salient-subject
    localization + geometric scoring). See
    app/quality/semantic/framing.py. All subject_* fields are None
    when no discernible subject was found (empty/near-empty saliency
    mask) — not a framing verdict in that case, just "not evaluated".
    """

    score: float
    detected: bool
    touches_edge: bool | None = None
    subject_position_score: float | None = None
    subject_area_ratio: float | None = None
    left_margin: float | None = None
    right_margin: float | None = None
    top_margin: float | None = None
    bottom_margin: float | None = None


class SuitabilityMetrics(BaseModel):
    """
    Final verdict from the trained Suitability model — combines every
    other detector's output into one suitable/not-suitable decision.
    See scripts/train_suitability_model.py for how it was trained.
    """

    suitable: bool
    # Model's predicted probability of the "suitable" class (0.0-1.0).
    confidence: float


class NotImplementedMetrics(BaseModel):
    status: str = "not_implemented"
    reason: str


class AssessmentResponse(BaseModel):
    filename: str
    blur: BlurMetrics
    exposure: ExposureMetrics
    glare: GlareMetrics | None = None
    motion: MotionMetrics
    resolution: ResolutionMetrics
    occlusion: OcclusionMetrics
    framing: FramingMetrics
    suitability: SuitabilityMetrics


class ErrorResponse(BaseModel):
    detail: str