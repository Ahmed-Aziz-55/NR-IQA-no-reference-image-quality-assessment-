import cv2
import numpy as np

# Laplacian and similar kernels need at least a 3x3 neighborhood.
MIN_DIMENSION = 3


def validate_image(image: np.ndarray, min_dimension: int = 1) -> None:
    """
    Validate that the input is a usable uint8 image array.

    Args:
        image: Candidate image array.
        min_dimension: Minimum required height/width in pixels.

    Raises:
        TypeError: If the input is not a NumPy array or is not uint8.
        ValueError: If the image is empty, too small, has an
            unsupported shape, or an unsupported channel count.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected a NumPy array, got {type(image).__name__}.")

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image shape: {image.shape}. "
            "Expected a 2D grayscale or 3D color image."
        )

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(
            f"Unsupported channel count: {image.shape[2]}. "
            "Expected 3 (BGR) or 4 (BGRA) channels."
        )

    height, width = image.shape[:2]

    if height < min_dimension or width < min_dimension:
        raise ValueError(
            f"Image too small ({height}x{width}). "
            f"Minimum required size is {min_dimension}x{min_dimension}."
        )

    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected dtype uint8, got {image.dtype}. "
            "Convert the image to uint8 (0-255 range) before processing."
        )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a validated BGR, BGRA, or grayscale image to grayscale.

    Args:
        image: Validated input image.

    Returns:
        Single-channel grayscale image.
    """
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
