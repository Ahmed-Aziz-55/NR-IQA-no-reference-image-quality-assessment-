import cv2
import numpy as np

from app.quality.classical._image_utils import MIN_DIMENSION, validate_image


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert a validated color image to 3-channel BGR.

    Args:
        image: Validated BGR or BGRA image.

    Returns:
        3-channel BGR image.

    Raises:
        ValueError: If the image has no color information (grayscale).
    """
    if image.ndim != 3:
        raise ValueError(
            "Glare detection requires a color (BGR/BGRA) image; "
            "grayscale images have no saturation channel."
        )

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _validate_hsv_thresholds(saturation_threshold: int, value_threshold: int) -> None:
    """
    Validate saturation and value thresholds are within the uint8 range.

    Raises:
        ValueError: If either threshold is outside 0-255.
    """
    if not 0 <= saturation_threshold <= 255:
        raise ValueError(
            f"saturation_threshold must be between 0 and 255, got {saturation_threshold}."
        )
    if not 0 <= value_threshold <= 255:
        raise ValueError(
            f"value_threshold must be between 0 and 255, got {value_threshold}."
        )


def calculate_glare_mask(
    image: np.ndarray,
    saturation_threshold: int = 60,
    value_threshold: int = 230,
) -> np.ndarray:
    """
    Compute a binary mask of candidate glare (specular highlight) pixels.

    A pixel is considered a glare candidate when it is both bright
    (high V) and washed-out/colorless (low S) in HSV space.

    Args:
        image: Input BGR or BGRA uint8 image.
        saturation_threshold: Pixels with saturation below this value
            are considered colorless.
        value_threshold: Pixels with brightness above this value
            are considered bright.

    Returns:
        A 2D uint8 mask (0 or 255) the same height/width as the input.

    Raises:
        ValueError: If the image is grayscale or thresholds are invalid.
    """
    validate_image(image, min_dimension=MIN_DIMENSION)
    _validate_hsv_thresholds(saturation_threshold, value_threshold)

    bgr = _to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    condition = (saturation < saturation_threshold) & (value > value_threshold)
    return (condition.astype(np.uint8)) * 255


def calculate_glare_area_ratio(
    image: np.ndarray,
    saturation_threshold: int = 60,
    value_threshold: int = 230,
) -> float:
    """
    Calculate the proportion of pixels flagged as glare candidates.

    Args:
        image: Input BGR or BGRA uint8 image.
        saturation_threshold: Saturation cutoff, see calculate_glare_mask.
        value_threshold: Brightness cutoff, see calculate_glare_mask.

    Returns:
        Ratio of glare-candidate pixels, from 0.0 to 1.0.
    """
    mask = calculate_glare_mask(image, saturation_threshold, value_threshold)
    return float(np.count_nonzero(mask) / mask.size)


def count_glare_regions(
    image: np.ndarray,
    saturation_threshold: int = 60,
    value_threshold: int = 230,
    min_region_area: int = 25,
) -> int:
    """
    Count distinct glare regions using connected component analysis.

    Small isolated bright/colorless pixels (sensor noise, specular
    dust, etc.) are filtered out via min_region_area so only
    meaningfully sized glare blobs are counted.

    Args:
        image: Input BGR or BGRA uint8 image.
        saturation_threshold: Saturation cutoff, see calculate_glare_mask.
        value_threshold: Brightness cutoff, see calculate_glare_mask.
        min_region_area: Minimum pixel area for a region to count as glare.

    Returns:
        Number of connected glare regions meeting the minimum area.

    Raises:
        ValueError: If min_region_area is not a positive integer.
    """
    if min_region_area < 1:
        raise ValueError(
            f"min_region_area must be a positive integer, got {min_region_area}."
        )

    mask = calculate_glare_mask(image, saturation_threshold, value_threshold)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Label 0 is always the background; skip it.
    region_areas = stats[1:num_labels, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(region_areas >= min_region_area))


def calculate_largest_glare_region_ratio(
    image: np.ndarray,
    saturation_threshold: int = 60,
    value_threshold: int = 230,
) -> float:
    """
    Calculate the area ratio of the single largest glare region.

    A large, contiguous glare blob is a stronger quality signal than
    many small scattered bright pixels; this metric isolates that case.

    Args:
        image: Input BGR or BGRA uint8 image.
        saturation_threshold: Saturation cutoff, see calculate_glare_mask.
        value_threshold: Brightness cutoff, see calculate_glare_mask.

    Returns:
        Ratio (0.0 to 1.0) of the largest glare region's area to the
        total image area. Returns 0.0 if no glare region exists.
    """
    mask = calculate_glare_mask(image, saturation_threshold, value_threshold)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return 0.0

    region_areas = stats[1:num_labels, cv2.CC_STAT_AREA]
    largest_area = int(np.max(region_areas))
    return float(largest_area / mask.size)
