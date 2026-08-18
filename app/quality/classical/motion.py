from dataclasses import dataclass

import cv2
import numpy as np

from app.quality.classical._image_utils import MIN_DIMENSION, to_grayscale, validate_image
from app.quality.classical.blur import calculate_laplacian_variance

# Small epsilon to avoid division by zero on perfectly flat/blank images,
# without producing literal inf/nan values that break downstream math.
_EPSILON = 1e-6


def _validate_ksize(ksize: int) -> None:
    """
    Validate the Sobel kernel size.

    Raises:
        ValueError: If ksize is not a positive odd integer >= 1.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")


def calculate_directional_gradient_variance(
    image: np.ndarray,
    ksize: int = 3,
) -> tuple[float, float]:
    """
    Calculate the variance of horizontal and vertical Sobel gradients.

    Motion blur suppresses gradient energy along the axis of motion,
    so comparing these two variances reveals directional blur.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        ksize: Aperture size for the Sobel kernel. Must be a
            positive odd integer (1, 3, 5, ...).

    Returns:
        A tuple (variance_x, variance_y) of the horizontal and
        vertical gradient variances.

    Raises:
        ValueError: If ksize is invalid or the image is too small.
    """
    validate_image(image, min_dimension=MIN_DIMENSION)
    _validate_ksize(ksize)

    gray = to_grayscale(image)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    return float(gradient_x.var()), float(gradient_y.var())


def calculate_motion_blur_ratio(image: np.ndarray, ksize: int = 3) -> float:
    """
    Calculate the directional gradient variance ratio.

    A ratio close to 1.0 indicates no dominant blur direction
    (sharp image or uniform/focus blur). A high ratio indicates
    a strong directional (motion) blur signature.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        ksize: Aperture size for the Sobel kernel, see
            calculate_directional_gradient_variance.

    Returns:
        Ratio of the larger to the smaller gradient variance,
        always >= 1.0.
    """
    variance_x, variance_y = calculate_directional_gradient_variance(image, ksize)

    numerator = max(variance_x, variance_y)
    denominator = min(variance_x, variance_y)

    # Epsilon smoothing keeps this finite even for perfectly flat images
    # (variance_x == variance_y == 0), instead of producing inf/nan.
    return float((numerator + _EPSILON) / (denominator + _EPSILON))


def calculate_motion_blur_direction(image: np.ndarray, ksize: int = 3) -> str:
    """
    Identify which axis carries the weaker gradient energy.

    The axis with lower gradient variance is the direction along
    which blur/smearing is most likely occurring.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        ksize: Aperture size for the Sobel kernel, see
            calculate_directional_gradient_variance.

    Returns:
        "horizontal" if horizontal gradients are weaker (suggesting
        horizontal motion smear), "vertical" if vertical gradients
        are weaker, or "none" if both are exactly equal.
    """
    variance_x, variance_y = calculate_directional_gradient_variance(image, ksize)

    if variance_x < variance_y:
        return "horizontal"
    if variance_y < variance_x:
        return "vertical"
    return "none"


@dataclass(frozen=True)
class MotionBlurAssessment:
    """
    Combined motion-blur signal.

    Attributes:
        ratio: Directional gradient variance ratio, see
            calculate_motion_blur_ratio.
        direction: Weaker-gradient axis, see calculate_motion_blur_direction.
        sharpness: Overall Laplacian variance of the image (both axes
            combined). Low values mean the image lacks detail generally.
        is_likely_motion_blur: True only when directional imbalance is
            paired with reduced overall sharpness. See rationale below.
    """

    ratio: float
    direction: str
    sharpness: float
    is_likely_motion_blur: bool


def assess_motion_blur(
    image: np.ndarray,
    ksize: int = 3,
    ratio_threshold: float = 3.0,
    sharpness_threshold: float = 150.0,
) -> MotionBlurAssessment:
    """
    Assess whether an image shows genuine motion blur, not just a
    directionally biased gradient ratio.

    Rationale: directional gradient anisotropy (calculate_motion_blur_ratio)
    is a NECESSARY but not SUFFICIENT signal for motion blur. A perfectly
    sharp image with strong one-directional content — sun rays, blinds,
    text lines, architectural verticals — produces a high ratio despite
    having zero blur. Treating "ratio > threshold" as "motion blurred" on
    its own therefore produces false positives on sharp, directionally
    textured images (this was observed directly: a sharp, well-lit image
    was still flagged with high confidence by ratio alone).

    This function only calls an image "likely motion blurred" when BOTH
    hold:
      1. The gradient ratio exceeds ratio_threshold (directional imbalance
         exists), AND
      2. Overall Laplacian variance (sharpness) is below sharpness_threshold
         (the image is actually short on high-frequency detail overall,
         consistent with smearing rather than sharp directional content).

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        ksize: Aperture size for the Sobel kernel.
        ratio_threshold: Minimum gradient ratio to consider directionally
            imbalanced. NOT calibrated against labeled data yet — treat as
            a placeholder pending docs/decision_engine.md calibration.
        sharpness_threshold: Laplacian variance below which the image is
            considered generally lacking detail. Same calibration caveat
            applies.

    Returns:
        MotionBlurAssessment with the raw signals and the combined verdict.
    """
    variance_x, variance_y = calculate_directional_gradient_variance(image, ksize)
    ratio = calculate_motion_blur_ratio(image, ksize)
    direction = calculate_motion_blur_direction(image, ksize)
    sharpness = calculate_laplacian_variance(image)

    is_likely_motion_blur = ratio >= ratio_threshold and sharpness < sharpness_threshold

    return MotionBlurAssessment(
        ratio=ratio,
        direction=direction,
        sharpness=sharpness,
        is_likely_motion_blur=is_likely_motion_blur,
    )
