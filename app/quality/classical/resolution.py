"""
Resolution / dimension checks.

IMPORTANT: the 640x480 defaults on is_below_minimum_resolution are a
generic placeholder, not a calibrated requirement. "Low resolution" is
only meaningful relative to what your downstream CV model actually
needs as input — a model that works fine at 320x320 should not reject
a 612x546 image just because it's below an arbitrary 640x480 cutoff.
Callers (including the API layer, see app/api/main.py's min_width/
min_height query params) should always pass the real requirement
explicitly rather than relying on the default.
"""

import numpy as np

from app.quality.classical._image_utils import validate_image


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """
    Get the width and height of an image.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.

    Returns:
        A tuple (width, height) in pixels.
    """
    validate_image(image)
    height, width = image.shape[:2]
    return width, height


def calculate_total_pixels(image: np.ndarray) -> int:
    """
    Calculate the total pixel count (width x height) of an image.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.

    Returns:
        Total number of pixels.
    """
    width, height = get_image_dimensions(image)
    return width * height


def calculate_aspect_ratio(image: np.ndarray) -> float:
    """
    Calculate the width-to-height aspect ratio of an image.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.

    Returns:
        Aspect ratio as width / height.
    """
    width, height = get_image_dimensions(image)
    return float(width / height)


def is_below_minimum_resolution(
    image: np.ndarray,
    min_width: int = 640,
    min_height: int = 480,
) -> bool:
    """
    Check whether an image falls below minimum resolution requirements.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        min_width: Minimum acceptable width in pixels.
        min_height: Minimum acceptable height in pixels.

    Returns:
        True if the image's width or height is below the given minimums.

    Raises:
        ValueError: If min_width or min_height is not a positive integer.
    """
    if min_width < 1 or min_height < 1:
        raise ValueError(
            f"min_width and min_height must be positive integers, "
            f"got min_width={min_width}, min_height={min_height}."
        )

    width, height = get_image_dimensions(image)
    return width < min_width or height < min_height
