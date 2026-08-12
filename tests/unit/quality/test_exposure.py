import numpy as np
import pytest

from app.quality.classical.exposure import (
    calculate_bright_pixel_ratio,
    calculate_dark_pixel_ratio,
    calculate_mean_brightness,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic images
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_mid_image() -> np.ndarray:
    """Uniform mid-gray grayscale image (intensity 128)."""
    return np.full((50, 50), 128, dtype=np.uint8)


@pytest.fixture
def bgr_mid_image() -> np.ndarray:
    """Uniform mid-gray BGR image (intensity 128 on all channels)."""
    return np.full((50, 50, 3), 128, dtype=np.uint8)


@pytest.fixture
def bgra_mid_image() -> np.ndarray:
    """Uniform mid-gray BGRA image (intensity 128, alpha 255)."""
    image = np.full((50, 50, 4), 128, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


@pytest.fixture
def pure_black_image() -> np.ndarray:
    """Fully black grayscale image (intensity 0)."""
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def pure_white_image() -> np.ndarray:
    """Fully white grayscale image (intensity 255)."""
    return np.full((50, 50), 255, dtype=np.uint8)


@pytest.fixture
def dark_image() -> np.ndarray:
    """Mostly dark image with a small bright patch."""
    image = np.full((100, 100), 10, dtype=np.uint8)
    image[0:10, 0:10] = 200
    return image


@pytest.fixture
def normal_image() -> np.ndarray:
    """Well-exposed image with a mid-range uniform intensity."""
    return np.full((100, 100), 120, dtype=np.uint8)


@pytest.fixture
def overexposed_image() -> np.ndarray:
    """Mostly bright/clipped image with a small dark patch."""
    image = np.full((100, 100), 250, dtype=np.uint8)
    image[0:10, 0:10] = 50
    return image


# ---------------------------------------------------------------------------
# calculate_mean_brightness — unit tests
# ---------------------------------------------------------------------------

def test_mean_brightness_grayscale(gray_mid_image):
    assert calculate_mean_brightness(gray_mid_image) == pytest.approx(128.0)


def test_mean_brightness_bgr(bgr_mid_image):
    assert calculate_mean_brightness(bgr_mid_image) == pytest.approx(128.0)


def test_mean_brightness_bgra(bgra_mid_image):
    assert calculate_mean_brightness(bgra_mid_image) == pytest.approx(128.0)


def test_mean_brightness_pure_black(pure_black_image):
    assert calculate_mean_brightness(pure_black_image) == pytest.approx(0.0)


def test_mean_brightness_pure_white(pure_white_image):
    assert calculate_mean_brightness(pure_white_image) == pytest.approx(255.0)


# ---------------------------------------------------------------------------
# calculate_mean_brightness — edge cases (shared validation contract)
# ---------------------------------------------------------------------------

def test_mean_brightness_raises_on_none():
    with pytest.raises(TypeError):
        calculate_mean_brightness(None)


def test_mean_brightness_raises_on_empty_array():
    with pytest.raises(ValueError):
        calculate_mean_brightness(np.array([], dtype=np.uint8))


def test_mean_brightness_raises_on_wrong_dtype():
    float_image = np.full((10, 10), 0.5, dtype=np.float32)
    with pytest.raises(TypeError):
        calculate_mean_brightness(float_image)


def test_mean_brightness_raises_on_invalid_channels():
    bad_image = np.zeros((10, 10, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_mean_brightness(bad_image)


# ---------------------------------------------------------------------------
# calculate_dark_pixel_ratio — unit tests
# ---------------------------------------------------------------------------

def test_dark_pixel_ratio_pure_black(pure_black_image):
    assert calculate_dark_pixel_ratio(pure_black_image, pixel_threshold=30) == pytest.approx(1.0)


def test_dark_pixel_ratio_pure_white(pure_white_image):
    assert calculate_dark_pixel_ratio(pure_white_image, pixel_threshold=30) == pytest.approx(0.0)


def test_dark_pixel_ratio_mixed_image(dark_image):
    ratio = calculate_dark_pixel_ratio(dark_image, pixel_threshold=30)
    # 100x100 image, 10x10 bright patch => 9900/10000 pixels are dark
    assert ratio == pytest.approx(0.99)


def test_dark_pixel_ratio_default_threshold(pure_black_image):
    assert calculate_dark_pixel_ratio(pure_black_image) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_dark_pixel_ratio — edge cases
# ---------------------------------------------------------------------------

def test_dark_pixel_ratio_raises_on_negative_threshold(gray_mid_image):
    with pytest.raises(ValueError):
        calculate_dark_pixel_ratio(gray_mid_image, pixel_threshold=-1)


def test_dark_pixel_ratio_raises_on_threshold_above_255(gray_mid_image):
    with pytest.raises(ValueError):
        calculate_dark_pixel_ratio(gray_mid_image, pixel_threshold=256)


def test_dark_pixel_ratio_threshold_boundary_is_exclusive():
    image = np.full((10, 10), 30, dtype=np.uint8)
    assert calculate_dark_pixel_ratio(image, pixel_threshold=30) == pytest.approx(0.0)


def test_dark_pixel_ratio_raises_on_invalid_image():
    with pytest.raises(TypeError):
        calculate_dark_pixel_ratio("not an image")


# ---------------------------------------------------------------------------
# calculate_bright_pixel_ratio — unit tests
# ---------------------------------------------------------------------------

def test_bright_pixel_ratio_pure_white(pure_white_image):
    assert calculate_bright_pixel_ratio(pure_white_image, pixel_threshold=245) == pytest.approx(1.0)


def test_bright_pixel_ratio_pure_black(pure_black_image):
    assert calculate_bright_pixel_ratio(pure_black_image, pixel_threshold=245) == pytest.approx(0.0)


def test_bright_pixel_ratio_mixed_image(overexposed_image):
    ratio = calculate_bright_pixel_ratio(overexposed_image, pixel_threshold=245)
    # 100x100 image, 10x10 dark patch => 9900/10000 pixels are bright
    assert ratio == pytest.approx(0.99)


def test_bright_pixel_ratio_default_threshold(pure_white_image):
    assert calculate_bright_pixel_ratio(pure_white_image) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_bright_pixel_ratio — edge cases
# ---------------------------------------------------------------------------

def test_bright_pixel_ratio_raises_on_negative_threshold(gray_mid_image):
    with pytest.raises(ValueError):
        calculate_bright_pixel_ratio(gray_mid_image, pixel_threshold=-5)


def test_bright_pixel_ratio_raises_on_threshold_above_255(gray_mid_image):
    with pytest.raises(ValueError):
        calculate_bright_pixel_ratio(gray_mid_image, pixel_threshold=300)


def test_bright_pixel_ratio_threshold_boundary_is_exclusive():
    image = np.full((10, 10), 245, dtype=np.uint8)
    assert calculate_bright_pixel_ratio(image, pixel_threshold=245) == pytest.approx(0.0)


def test_bright_pixel_ratio_raises_on_invalid_image():
    with pytest.raises(TypeError):
        calculate_bright_pixel_ratio(None)


# ---------------------------------------------------------------------------
# Behavior tests — dark vs normal, overexposed vs normal
# ---------------------------------------------------------------------------

def test_dark_image_has_lower_brightness_than_normal(dark_image, normal_image):
    dark_brightness = calculate_mean_brightness(dark_image)
    normal_brightness = calculate_mean_brightness(normal_image)
    assert dark_brightness < normal_brightness


def test_dark_image_has_higher_dark_ratio_than_normal(dark_image, normal_image):
    dark_ratio = calculate_dark_pixel_ratio(dark_image, pixel_threshold=30)
    normal_ratio = calculate_dark_pixel_ratio(normal_image, pixel_threshold=30)
    assert dark_ratio > normal_ratio


def test_overexposed_image_has_higher_brightness_than_normal(overexposed_image, normal_image):
    overexposed_brightness = calculate_mean_brightness(overexposed_image)
    normal_brightness = calculate_mean_brightness(normal_image)
    assert overexposed_brightness > normal_brightness


def test_overexposed_image_has_higher_bright_ratio_than_normal(overexposed_image, normal_image):
    overexposed_ratio = calculate_bright_pixel_ratio(overexposed_image, pixel_threshold=245)
    normal_ratio = calculate_bright_pixel_ratio(normal_image, pixel_threshold=245)
    assert overexposed_ratio > normal_ratio


def test_normal_image_has_low_dark_and_bright_ratios(normal_image):
    dark_ratio = calculate_dark_pixel_ratio(normal_image, pixel_threshold=30)
    bright_ratio = calculate_bright_pixel_ratio(normal_image, pixel_threshold=245)
    assert dark_ratio == pytest.approx(0.0)
    assert bright_ratio == pytest.approx(0.0)
