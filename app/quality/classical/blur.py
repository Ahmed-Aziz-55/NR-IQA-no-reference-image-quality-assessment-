import cv2
import numpy as np

from app.quality.classical._image_utils import (
    MIN_DIMENSION,
    to_grayscale,
    validate_image,
)


def calculate_laplacian_variance(image: np.ndarray, ksize: int = 1) -> float:
    """
    Calculate the variance of the Laplacian response.

    Higher values generally indicate stronger image detail,
    while lower values generally indicate reduced sharpness.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        ksize: Aperture size for the Laplacian kernel. Must be a
            positive odd integer (1, 3, 5, ...).

    Returns:
        Laplacian variance as a float.

    Raises:
        ValueError: If ksize is invalid or the image is too small.
    """
    validate_image(image, min_dimension=MIN_DIMENSION)

    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")

    gray = to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

    return float(laplacian.var())
