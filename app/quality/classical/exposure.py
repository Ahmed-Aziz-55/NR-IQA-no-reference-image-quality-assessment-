import numpy as np

from app.quality.classical._image_utils import to_grayscale, validate_image


def calculate_mean_brightness(image: np.ndarray) -> float:
    """
    Calculate the mean grayscale brightness of an image.

    The returned value is the raw mean intensity in the range 0 to 255.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.

    Returns:
        Mean grayscale intensity as a float.
    """
    validate_image(image)
    gray = to_grayscale(image)
    return float(np.mean(gray))


def calculate_dark_pixel_ratio(
    image: np.ndarray,
    pixel_threshold: int = 30,
) -> float:
    """
    Calculate the proportion of pixels below a brightness threshold.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        pixel_threshold: Pixels below this grayscale intensity
            are considered dark.

    Returns:
        Ratio of dark pixels in the image, from 0.0 to 1.0.

    Raises:
        ValueError: If pixel_threshold is outside the uint8 range.
    """
    validate_image(image)
    if not 0 <= pixel_threshold <= 255:
        raise ValueError(
            f"pixel_threshold must be between 0 and 255, got {pixel_threshold}."
        )
    gray = to_grayscale(image)
    dark_pixels = np.count_nonzero(gray < pixel_threshold)
    return float(dark_pixels / gray.size)


def calculate_bright_pixel_ratio(
    image: np.ndarray,
    pixel_threshold: int = 245,
) -> float:
    """
    Calculate the proportion of pixels above a brightness threshold.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        pixel_threshold: Pixels above this grayscale intensity
            are considered bright.

    Returns:
        Ratio of bright pixels in the image, from 0.0 to 1.0.

    Raises:
        ValueError: If pixel_threshold is outside the uint8 range.
    """
    validate_image(image)
    if not 0 <= pixel_threshold <= 255:
        raise ValueError(
            f"pixel_threshold must be between 0 and 255, got {pixel_threshold}."
        )
    gray = to_grayscale(image)
    bright_pixels = np.count_nonzero(gray > pixel_threshold)
    return float(bright_pixels / gray.size)


def calculate_saturated_pixel_ratio(
    image: np.ndarray,
    pixel_threshold: int = 250,
) -> float:
    """
    Calculate the proportion of pixels that are clipped/saturated.

    Distinct from calculate_bright_pixel_ratio: "bright" (>245 by default)
    covers a wide, soft range and can be high for an image that is simply
    well-lit and light-toned. "Saturated" uses a tighter, near-255 cutoff
    intended to isolate true sensor clipping — detail that has been lost
    because the sensor hit its maximum value, not just a light-colored
    subject. A well-lit image can have a high bright_pixel_ratio with a
    near-zero saturated_pixel_ratio; that combination indicates "bright but
    not overexposed", which mean brightness or bright_pixel_ratio alone
    cannot distinguish.

    Args:
        image: Input BGR, BGRA, or grayscale uint8 image.
        pixel_threshold: Pixels at or above this grayscale intensity are
            considered clipped. Defaults to 250, tighter than
            calculate_bright_pixel_ratio's default of 245.

    Returns:
        Ratio of saturated pixels in the image, from 0.0 to 1.0.

    Raises:
        ValueError: If pixel_threshold is outside the uint8 range.
    """
    validate_image(image)
    if not 0 <= pixel_threshold <= 255:
        raise ValueError(
            f"pixel_threshold must be between 0 and 255, got {pixel_threshold}."
        )
    gray = to_grayscale(image)
    saturated_pixels = np.count_nonzero(gray >= pixel_threshold)
    return float(saturated_pixels / gray.size)
