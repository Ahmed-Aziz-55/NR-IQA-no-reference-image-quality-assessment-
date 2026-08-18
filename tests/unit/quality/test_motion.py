import cv2
import numpy as np
import pytest

from app.quality.classical.motion import (
    calculate_directional_gradient_variance,
    calculate_motion_blur_direction,
    calculate_motion_blur_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic images
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_image() -> np.ndarray:
    """Perfectly flat image — no gradients in either direction."""
    return np.full((100, 100), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """Checkerboard pattern — roughly equal gradient energy in both axes."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[::2, ::2] = 255
    image[1::2, 1::2] = 255
    return image


@pytest.fixture
def vertical_stripes_image() -> np.ndarray:
    """
    Vertical stripes (alternating columns). Sharp transitions occur
    along the x-axis, so this produces strong horizontal gradients
    (high Var(Gx)) and near-zero vertical gradients (low Var(Gy)).
    """
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, ::4] = 255
    image[:, 1::4] = 255
    return image


@pytest.fixture
def horizontally_motion_blurred_image(vertical_stripes_image) -> np.ndarray:
    """
    Apply a horizontal motion-blur kernel to the vertical stripes image.
    This smears content along the x-axis, suppressing Var(Gx) while
    leaving Var(Gy) close to zero as well — simulating horizontal smear.
    """
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(vertical_stripes_image, -1, kernel)


@pytest.fixture
def bgr_checkerboard_image(checkerboard_image) -> np.ndarray:
    """BGR version of the checkerboard image to confirm channel handling."""
    return cv2.cvtColor(checkerboard_image, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# calculate_directional_gradient_variance — unit tests
# ---------------------------------------------------------------------------

def test_directional_variance_flat_image_is_zero(flat_image):
    variance_x, variance_y = calculate_directional_gradient_variance(flat_image)
    assert variance_x == pytest.approx(0.0)
    assert variance_y == pytest.approx(0.0)


def test_directional_variance_bgr_matches_grayscale(checkerboard_image, bgr_checkerboard_image):
    gray_result = calculate_directional_gradient_variance(checkerboard_image)
    bgr_result = calculate_directional_gradient_variance(bgr_checkerboard_image)
    assert gray_result == pytest.approx(bgr_result)


def test_directional_variance_vertical_stripes_favors_x(vertical_stripes_image):
    variance_x, variance_y = calculate_directional_gradient_variance(vertical_stripes_image)
    assert variance_x > variance_y


# ---------------------------------------------------------------------------
# calculate_directional_gradient_variance — edge cases
# ---------------------------------------------------------------------------

def test_directional_variance_raises_on_invalid_ksize(flat_image):
    with pytest.raises(ValueError):
        calculate_directional_gradient_variance(flat_image, ksize=4)


def test_directional_variance_raises_on_non_numpy_input():
    with pytest.raises(TypeError):
        calculate_directional_gradient_variance("not an image")


def test_directional_variance_raises_on_too_small_image():
    tiny_image = np.full((1, 1), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_directional_gradient_variance(tiny_image)


# ---------------------------------------------------------------------------
# calculate_motion_blur_ratio — unit tests
# ---------------------------------------------------------------------------

def test_motion_blur_ratio_flat_image_is_near_one(flat_image):
    # Both variances are 0 -> epsilon smoothing yields a ratio of ~1.0.
    assert calculate_motion_blur_ratio(flat_image) == pytest.approx(1.0)


def test_motion_blur_ratio_is_always_at_least_one(checkerboard_image):
    ratio = calculate_motion_blur_ratio(checkerboard_image)
    assert ratio >= 1.0


def test_motion_blur_ratio_high_for_directional_pattern(vertical_stripes_image):
    ratio = calculate_motion_blur_ratio(vertical_stripes_image)
    assert ratio > 10.0  # strong directional bias expected


# ---------------------------------------------------------------------------
# calculate_motion_blur_ratio — edge cases
# ---------------------------------------------------------------------------

def test_motion_blur_ratio_raises_on_invalid_ksize(flat_image):
    with pytest.raises(ValueError):
        calculate_motion_blur_ratio(flat_image, ksize=0)


def test_motion_blur_ratio_raises_on_empty_image():
    with pytest.raises(ValueError):
        calculate_motion_blur_ratio(np.array([], dtype=np.uint8))


# ---------------------------------------------------------------------------
# calculate_motion_blur_direction — unit tests
# ---------------------------------------------------------------------------

def test_motion_blur_direction_flat_image_is_none(flat_image):
    assert calculate_motion_blur_direction(flat_image) == "none"


def test_motion_blur_direction_vertical_stripes_is_vertical(vertical_stripes_image):
    # Var(Gy) is the weaker axis for vertical-stripe content,
    # so the detected smear direction is "vertical".
    assert calculate_motion_blur_direction(vertical_stripes_image) == "vertical"


# ---------------------------------------------------------------------------
# Behavior test — motion-blurred vs sharp directional pattern
# ---------------------------------------------------------------------------

def test_motion_blurred_image_has_lower_x_variance_than_sharp(
    vertical_stripes_image, horizontally_motion_blurred_image
):
    sharp_variance_x, _ = calculate_directional_gradient_variance(vertical_stripes_image)
    blurred_variance_x, _ = calculate_directional_gradient_variance(
        horizontally_motion_blurred_image
    )
    assert blurred_variance_x < sharp_variance_x
