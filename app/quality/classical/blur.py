import cv2
import numpy as np

_MIN_DIMENSION = 3


def _validate_image(image: np.ndarray) -> None:
    """
    Validate that the input is a usable image array.

    Args:
        image: Candidate image array.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the image is empty, too small, or has an
            unsupported number of channels.
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

    if height < _MIN_DIMENSION or width < _MIN_DIMENSION:
        raise ValueError(
            f"Image too small ({height}x{width}). "
            f"Minimum required size is {_MIN_DIMENSION}x{_MIN_DIMENSION}."
        )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a validated BGR, BGRA, or grayscale image to grayscale.
    """
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_laplacian_variance(
    image: np.ndarray,
    ksize: int = 1,
) -> float:
    """
    Calculate the variance of the Laplacian response.

    Higher values generally indicate stronger image detail,
    while lower values generally indicate reduced sharpness.

    Args:
        image: Input image as a BGR, BGRA, or grayscale NumPy array.
        ksize: Aperture size for the Laplacian kernel. Must be a
            positive odd integer.

    Returns:
        Laplacian variance as a float.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the image is invalid or ksize is invalid.
    """
    _validate_image(image)

    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(
            f"ksize must be a positive odd integer, got {ksize}."
        )

    gray = _to_grayscale(image)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
        ksize=ksize,
    )

    return float(laplacian.var())
