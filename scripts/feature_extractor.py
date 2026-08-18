"""
Runs every implemented production detector against a single image
and flattens the results into one feature dict for the Suitability model.

The same detector implementations used by the production API are used
here so training and inference receive the same signals.
"""

from dataclasses import dataclass

import numpy as np

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
from app.quality.semantic.occlusion import ModelOcclusionDetector
from app.quality.semantic.framing import ModelFramingDetector


@dataclass
class FeatureExtractor:
    occlusion_detector: ModelOcclusionDetector
    framing_detector: ModelFramingDetector

    min_width: int = 640
    min_height: int = 480

    def extract(self, image: np.ndarray) -> dict[str, float]:
        features: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Blur
        # ------------------------------------------------------------------

        features["blur_laplacian_variance"] = (
            calculate_laplacian_variance(image)
        )

        # ------------------------------------------------------------------
        # Exposure
        # ------------------------------------------------------------------

        features["exposure_mean_brightness"] = (
            calculate_mean_brightness(image)
        )

        features["exposure_dark_pixel_ratio"] = (
            calculate_dark_pixel_ratio(image)
        )

        features["exposure_bright_pixel_ratio"] = (
            calculate_bright_pixel_ratio(image)
        )

        features["exposure_saturated_pixel_ratio"] = (
            calculate_saturated_pixel_ratio(image)
        )

        # ------------------------------------------------------------------
        # Glare
        # ------------------------------------------------------------------

        if image.ndim == 3:
            features["glare_area_ratio"] = (
                calculate_glare_area_ratio(image)
            )

            features["glare_region_count"] = float(
                count_glare_regions(image)
            )

            features["glare_largest_region_ratio"] = (
                calculate_largest_glare_region_ratio(image)
            )
        else:
            features["glare_area_ratio"] = 0.0
            features["glare_region_count"] = 0.0
            features["glare_largest_region_ratio"] = 0.0

        # ------------------------------------------------------------------
        # Motion
        # ------------------------------------------------------------------

        motion_assessment = assess_motion_blur(image)

        variance_x, variance_y = (
            calculate_directional_gradient_variance(image)
        )

        features["motion_gradient_variance_x"] = variance_x
        features["motion_gradient_variance_y"] = variance_y
        features["motion_blur_ratio"] = motion_assessment.ratio
        features["motion_overall_sharpness"] = motion_assessment.sharpness
        features["motion_is_likely_motion_blur"] = float(
            motion_assessment.is_likely_motion_blur
        )

        # ------------------------------------------------------------------
        # Resolution
        # ------------------------------------------------------------------

        width, height = get_image_dimensions(image)

        features["resolution_width"] = float(width)
        features["resolution_height"] = float(height)

        features["resolution_total_pixels"] = float(
            calculate_total_pixels(image)
        )

        features["resolution_aspect_ratio"] = (
            calculate_aspect_ratio(image)
        )

        features["resolution_is_below_minimum"] = float(
            is_below_minimum_resolution(
                image,
                min_width=self.min_width,
                min_height=self.min_height,
            )
        )

        # ------------------------------------------------------------------
        # Occlusion
        # ------------------------------------------------------------------

        if image.ndim == 3:
            occlusion_result = self.occlusion_detector.assess(image)

            features["occlusion_score"] = float(
                occlusion_result.score
            )

            features["occlusion_detected"] = float(
                occlusion_result.detected
            )

            features["occlusion_face_detected"] = float(
                bool(occlusion_result.face_detected)
            )

            # IMPORTANT:
            # A face-based occlusion detector is applicable only when
            # a face was detected.
            features["occlusion_applicable"] = float(
                bool(occlusion_result.face_detected)
            )

        else:
            features["occlusion_score"] = 1.0
            features["occlusion_detected"] = 0.0
            features["occlusion_face_detected"] = 0.0
            features["occlusion_applicable"] = 0.0

        # ------------------------------------------------------------------
        # Framing
        # ------------------------------------------------------------------

        if image.ndim == 3:
            framing_result = self.framing_detector.assess(image)

            features["framing_score"] = float(
                framing_result.score
            )

            features["framing_detected"] = float(
                framing_result.detected
            )

            # Framing is applicable only when the detector successfully
            # localized a subject.
            #
            # subject_area_ratio is expected to be None when no reliable
            # subject was found.
            features["framing_applicable"] = float(
                framing_result.subject_area_ratio is not None
            )

        else:
            # No reliable framing analysis for grayscale input.
            features["framing_score"] = 0.5
            features["framing_detected"] = 0.0
            features["framing_applicable"] = 0.0

        return features