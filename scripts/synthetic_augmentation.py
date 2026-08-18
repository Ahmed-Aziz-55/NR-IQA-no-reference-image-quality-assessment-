"""
Injects controlled, severity-graded synthetic defects into clean
images, to generate ground-truth labels for the Suitability model
that are INDEPENDENT of the classical detectors themselves (the
detectors' outputs become the model's input FEATURES; the injected
severity becomes the LABEL — keeping training uncontaminated by the
very detectors being evaluated).

Severity levels: none / low / medium / high, per defect type.

Darkness and Overexposure are treated as mutually exclusive (both are
opposite ends of the same brightness axis — an image can't be both
injected-dark and injected-bright at once), via sample_exposure_severity().
Every other defect type is sampled independently and can co-occur.

Resolution works differently from the rest: instead of degrading pixel
content, it actually resizes the image to different real dimensions
(preserving KonIQ's 4:3 aspect ratio), since the Resolution detector
measures actual width/height, not sharpness/content.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

SEVERITY_LEVELS = ("none", "low", "medium", "high")

# Target (width, height) per resolution severity level. Chosen to keep
# a 4:3 aspect ratio matching KonIQ-10k's native 512x384, and to
# straddle the API's default 640x480 minimum-resolution threshold:
# "none"/"low" land at or above it, "medium"/"high" fall below.
_RESOLUTION_TARGETS = {
    "none": (1024, 768),
    "low": (800, 600),
    "medium": (480, 360),
    "high": (240, 180),
}

_DARKNESS_FACTORS = {"low": 0.70, "medium": 0.45, "high": 0.20}
_OVEREXPOSURE_FACTORS = {"low": 1.3, "medium": 1.7, "high": 2.4}
_BLUR_KERNELS = {"low": 5, "medium": 11, "high": 21}
_MOTION_BLUR_LENGTHS = {"low": 7, "medium": 15, "high": 25}
_GLARE_AREA_FRACTIONS = {"low": 0.03, "medium": 0.08, "high": 0.18}
_GLARE_ALPHAS = {"low": 0.5, "medium": 0.75, "high": 0.95}

# Real-world occlusion severity tiers, derived from COFW's continuous
# occluded_landmark_ratio (see extract_cofw.py) rather than injected —
# occlusion in this dataset comes from genuine hand/object occlusion,
# not a synthetic overlay.
_OCCLUSION_RATIO_TIERS = (
    (0.0, "none"),
    (0.15, "low"),
    (0.35, "medium"),
)


def occlusion_severity_from_ratio(occluded_landmark_ratio: float) -> str:
    """Map COFW's continuous occlusion ratio to a severity tier."""
    if occluded_landmark_ratio <= 0.0:
        return "none"
    if occluded_landmark_ratio < 0.15:
        return "low"
    if occluded_landmark_ratio < 0.35:
        return "medium"
    return "high"


@dataclass(frozen=True)
class InjectedSeverities:
    """The ground-truth severity assigned per defect type for one sample."""

    blur: str = "none"
    darkness: str = "none"
    overexposure: str = "none"
    glare: str = "none"
    motion: str = "none"
    resolution: str = "none"
    occlusion: str = "none"

    def as_dict(self) -> dict[str, str]:
        return {
            "blur": self.blur,
            "darkness": self.darkness,
            "overexposure": self.overexposure,
            "glare": self.glare,
            "motion": self.motion,
            "resolution": self.resolution,
            "occlusion": self.occlusion,
        }

    def is_suitable(self) -> bool:
        """
        Label rule (project decision): NOT suitable if any defect is
        "high" severity, OR if 2+ defects are "medium" severity.
        Suitable otherwise.
        """
        values = self.as_dict().values()
        if any(v == "high" for v in values):
            return False
        if sum(1 for v in values if v == "medium") >= 2:
            return False
        return True


def sample_exposure_severity(rng: np.random.Generator) -> tuple[str, str]:
    """
    Returns (darkness_severity, overexposure_severity) — mutually
    exclusive; at most one is non-"none".
    """
    choice = rng.choice(["none", "darkness", "overexposure"], p=[0.4, 0.3, 0.3])
    if choice == "none":
        return "none", "none"

    level = rng.choice(["low", "medium", "high"], p=[0.4, 0.35, 0.25])
    if choice == "darkness":
        return level, "none"
    return "none", level


def sample_severity(
    rng: np.random.Generator,
    p_none: float = 0.45,
    p_low: float = 0.25,
    p_medium: float = 0.2,
    p_high: float = 0.1,
) -> str:
    return rng.choice(["none", "low", "medium", "high"], p=[p_none, p_low, p_medium, p_high])


def apply_darkness(image: np.ndarray, severity: str) -> np.ndarray:
    if severity == "none":
        return image
    factor = _DARKNESS_FACTORS[severity]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, severity: str) -> np.ndarray:
    if severity == "none":
        return image
    factor = _OVEREXPOSURE_FACTORS[severity]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, severity: str) -> np.ndarray:
    if severity == "none":
        return image
    k = _BLUR_KERNELS[severity]
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_motion_blur(image: np.ndarray, severity: str, rng: np.random.Generator) -> np.ndarray:
    if severity == "none":
        return image
    length = _MOTION_BLUR_LENGTHS[severity]
    angle = float(rng.uniform(0, 180))

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return cv2.filter2D(image, -1, kernel)


def apply_glare(image: np.ndarray, severity: str, rng: np.random.Generator) -> np.ndarray:
    if severity == "none":
        return image

    height, width = image.shape[:2]
    area_fraction = _GLARE_AREA_FRACTIONS[severity]
    radius = max(1, int(np.sqrt(area_fraction * height * width / np.pi)))

    cx = int(rng.integers(radius, max(radius + 1, width - radius)))
    cy = int(rng.integers(radius, max(radius + 1, height - radius)))

    overlay = image.copy()
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius / 3)

    alpha = _GLARE_ALPHAS[severity]
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def apply_resolution(image: np.ndarray, severity: str) -> np.ndarray:
    target_w, target_h = _RESOLUTION_TARGETS[severity]
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


@dataclass
class SyntheticAugmentationEngine:
    """
    Applies a randomly (or explicitly) sampled combination of defect
    severities to a clean image, returning the degraded image plus the
    ground-truth InjectedSeverities used to produce it.
    """

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def sample_severities(self, occlusion_severity: str = "none") -> InjectedSeverities:
        darkness, overexposure = sample_exposure_severity(self.rng)
        return InjectedSeverities(
            blur=sample_severity(self.rng),
            darkness=darkness,
            overexposure=overexposure,
            glare=sample_severity(self.rng),
            motion=sample_severity(self.rng),
            resolution=sample_severity(self.rng),
            occlusion=occlusion_severity,
        )

    def apply(self, image: np.ndarray, severities: InjectedSeverities) -> np.ndarray:
        """
        Applies every non-occlusion defect (occlusion is never
        synthetically injected — see occlusion_severity_from_ratio,
        it comes from real COFW annotations instead). Order: exposure
        adjustments first, then blur/motion, then glare, then
        resolution last (resolution changes the actual pixel
        dimensions, so it goes after content-level edits).
        """
        result = image
        result = apply_darkness(result, severities.darkness)
        result = apply_overexposure(result, severities.overexposure)
        result = apply_blur(result, severities.blur)
        result = apply_motion_blur(result, severities.motion, self.rng)
        result = apply_glare(result, severities.glare, self.rng)
        result = apply_resolution(result, severities.resolution)
        return result