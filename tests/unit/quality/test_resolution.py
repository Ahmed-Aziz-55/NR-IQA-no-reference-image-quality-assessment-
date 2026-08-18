import numpy as np
import pytest

from app.quality.classical.resolution import (
    calculate_aspect_ratio,
    calculate_total_pixels,
    get_image_dimensions,
    is_below_minimum_resolution,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic images
# ---------------------------------------------------------------------------

@pytest.fixture
def hd_image() -> np.ndarray:
    """1280x720 grayscale image (width=1280, height=720)."""
    return np.zeros((720, 1280), dtype=np.uint8)


@pytest.fixture
def low_res_image() -> np.ndarray:
    """160x120 grayscale image — well below typical minimums."""
    return np.zeros((120, 160), dtype=np.uint8)


@pytest.fixture
def square_bgr_image() -> np.ndarray:
    """100x100 BGR image for aspect ratio = 1.0 checks."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# get_image_dimensions — unit tests
# ---------------------------------------------------------------------------

def test_get_image_dimensions_returns_width_height(hd_image):
    width, height = get_image_dimensions(hd_image)
    assert width == 1280
    assert height == 720


def test_get_image_dimensions_handles_color_image(square_bgr_image):
    width, height = get_image_dimensions(square_bgr_image)
    assert width == 100
    assert height == 100


# ---------------------------------------------------------------------------
# get_image_dimensions — edge cases
# ---------------------------------------------------------------------------

def test_get_image_dimensions_raises_on_non_numpy_input():
    with pytest.raises(TypeError):
        get_image_dimensions("not an image")


def test_get_image_dimensions_raises_on_empty_image():
    with pytest.raises(ValueError):
        get_image_dimensions(np.array([], dtype=np.uint8))


# ---------------------------------------------------------------------------
# calculate_total_pixels — unit tests
# ---------------------------------------------------------------------------

def test_calculate_total_pixels_hd_image(hd_image):
    assert calculate_total_pixels(hd_image) == 1280 * 720


def test_calculate_total_pixels_low_res_image(low_res_image):
    assert calculate_total_pixels(low_res_image) == 160 * 120


# ---------------------------------------------------------------------------
# calculate_aspect_ratio — unit tests
# ---------------------------------------------------------------------------

def test_calculate_aspect_ratio_square_image(square_bgr_image):
    assert calculate_aspect_ratio(square_bgr_image) == pytest.approx(1.0)


def test_calculate_aspect_ratio_widescreen_image(hd_image):
    assert calculate_aspect_ratio(hd_image) == pytest.approx(1280 / 720)


# ---------------------------------------------------------------------------
# is_below_minimum_resolution — unit tests
# ---------------------------------------------------------------------------

def test_is_below_minimum_resolution_true_for_low_res(low_res_image):
    assert is_below_minimum_resolution(low_res_image, min_width=640, min_height=480) is True


def test_is_below_minimum_resolution_false_for_hd(hd_image):
    assert is_below_minimum_resolution(hd_image, min_width=640, min_height=480) is False


def test_is_below_minimum_resolution_true_when_only_width_fails():
    image = np.zeros((720, 500), dtype=np.uint8)  # width below min, height fine
    assert is_below_minimum_resolution(image, min_width=640, min_height=480) is True


def test_is_below_minimum_resolution_true_when_only_height_fails():
    image = np.zeros((300, 1280), dtype=np.uint8)  # height below min, width fine
    assert is_below_minimum_resolution(image, min_width=640, min_height=480) is True


def test_is_below_minimum_resolution_boundary_is_inclusive_pass():
    # Exactly at the minimum should NOT count as "below" (strict '<').
    image = np.zeros((480, 640), dtype=np.uint8)
    assert is_below_minimum_resolution(image, min_width=640, min_height=480) is False


# ---------------------------------------------------------------------------
# is_below_minimum_resolution — edge cases
# ---------------------------------------------------------------------------

def test_is_below_minimum_resolution_raises_on_invalid_min_width(hd_image):
    with pytest.raises(ValueError):
        is_below_minimum_resolution(hd_image, min_width=0, min_height=480)


def test_is_below_minimum_resolution_raises_on_invalid_min_height(hd_image):
    with pytest.raises(ValueError):
        is_below_minimum_resolution(hd_image, min_width=640, min_height=-10)


# ---------------------------------------------------------------------------
# Behavior test — low-res vs high-res
# ---------------------------------------------------------------------------

def test_low_res_image_has_fewer_total_pixels_than_hd(low_res_image, hd_image):
    assert calculate_total_pixels(low_res_image) < calculate_total_pixels(hd_image)
