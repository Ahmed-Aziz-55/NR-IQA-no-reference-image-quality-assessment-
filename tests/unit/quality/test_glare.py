import cv2
import numpy as np
import pytest

from app.quality.classical.glare import (
    calculate_glare_area_ratio,
    calculate_glare_mask,
    calculate_largest_glare_region_ratio,
    count_glare_regions,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic images
# ---------------------------------------------------------------------------

@pytest.fixture
def colored_image_no_glare() -> np.ndarray:
    """Uniform saturated mid-tone BGR image with no glare."""
    return np.full((100, 100, 3), (120, 80, 40), dtype=np.uint8)


@pytest.fixture
def colored_image_with_one_glare_spot() -> np.ndarray:
    """Colored image with a single bright, low-saturation patch."""
    image = np.full((100, 100, 3), (120, 80, 40), dtype=np.uint8)
    image[10:20, 10:20] = (255, 255, 255)  # 10x10 white glare spot
    return image


@pytest.fixture
def colored_image_with_two_glare_spots() -> np.ndarray:
    """Colored image with two separate bright, low-saturation patches."""
    image = np.full((100, 100, 3), (120, 80, 40), dtype=np.uint8)
    image[5:15, 5:15] = (255, 255, 255)
    image[80:90, 80:90] = (250, 250, 250)
    return image


@pytest.fixture
def bgra_image_with_glare() -> np.ndarray:
    """BGRA variant with a glare spot to confirm alpha channel is handled."""
    image = np.full((100, 100, 4), (120, 80, 40, 255), dtype=np.uint8)
    image[10:20, 10:20] = (255, 255, 255, 255)
    return image


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Grayscale image — glare detection should reject this."""
    return np.full((50, 50), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# calculate_glare_mask — unit tests
# ---------------------------------------------------------------------------

def test_glare_mask_shape_matches_image(colored_image_with_one_glare_spot):
    mask = calculate_glare_mask(colored_image_with_one_glare_spot)
    assert mask.shape == colored_image_with_one_glare_spot.shape[:2]


def test_glare_mask_flags_white_patch(colored_image_with_one_glare_spot):
    mask = calculate_glare_mask(colored_image_with_one_glare_spot)
    assert mask[15, 15] == 255  # inside the white patch


def test_glare_mask_ignores_saturated_background(colored_image_no_glare):
    mask = calculate_glare_mask(colored_image_no_glare)
    assert np.count_nonzero(mask) == 0


def test_glare_mask_handles_bgra(bgra_image_with_glare):
    mask = calculate_glare_mask(bgra_image_with_glare)
    assert mask[15, 15] == 255


def test_glare_mask_raises_on_grayscale(grayscale_image):
    with pytest.raises(ValueError):
        calculate_glare_mask(grayscale_image)


# ---------------------------------------------------------------------------
# calculate_glare_mask — edge cases
# ---------------------------------------------------------------------------

def test_glare_mask_raises_on_invalid_saturation_threshold(colored_image_no_glare):
    with pytest.raises(ValueError):
        calculate_glare_mask(colored_image_no_glare, saturation_threshold=-1)


def test_glare_mask_raises_on_invalid_value_threshold(colored_image_no_glare):
    with pytest.raises(ValueError):
        calculate_glare_mask(colored_image_no_glare, value_threshold=300)


def test_glare_mask_raises_on_non_numpy_input():
    with pytest.raises(TypeError):
        calculate_glare_mask("not an image")


# ---------------------------------------------------------------------------
# calculate_glare_area_ratio — unit tests
# ---------------------------------------------------------------------------

def test_glare_area_ratio_no_glare(colored_image_no_glare):
    assert calculate_glare_area_ratio(colored_image_no_glare) == pytest.approx(0.0)


def test_glare_area_ratio_matches_patch_size(colored_image_with_one_glare_spot):
    # 100x100 image, 10x10 glare patch => 100/10000
    ratio = calculate_glare_area_ratio(colored_image_with_one_glare_spot)
    assert ratio == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# count_glare_regions — unit tests
# ---------------------------------------------------------------------------

def test_count_glare_regions_none(colored_image_no_glare):
    assert count_glare_regions(colored_image_no_glare) == 0


def test_count_glare_regions_single_spot(colored_image_with_one_glare_spot):
    assert count_glare_regions(colored_image_with_one_glare_spot) == 1


def test_count_glare_regions_two_spots(colored_image_with_two_glare_spots):
    assert count_glare_regions(colored_image_with_two_glare_spots) == 2


def test_count_glare_regions_filters_small_noise(colored_image_no_glare):
    # Add a single-pixel bright/colorless dot — should be filtered out
    # by the default min_region_area (25).
    image = colored_image_no_glare.copy()
    image[0, 0] = (255, 255, 255)
    assert count_glare_regions(image, min_region_area=25) == 0
    assert count_glare_regions(image, min_region_area=1) == 1


def test_count_glare_regions_raises_on_invalid_min_area(colored_image_no_glare):
    with pytest.raises(ValueError):
        count_glare_regions(colored_image_no_glare, min_region_area=0)


# ---------------------------------------------------------------------------
# calculate_largest_glare_region_ratio — unit tests
# ---------------------------------------------------------------------------

def test_largest_glare_region_ratio_no_glare(colored_image_no_glare):
    assert calculate_largest_glare_region_ratio(colored_image_no_glare) == pytest.approx(0.0)


def test_largest_glare_region_ratio_picks_bigger_spot():
    image = np.full((100, 100, 3), (120, 80, 40), dtype=np.uint8)
    image[0:5, 0:5] = (255, 255, 255)      # 25 px small spot
    image[50:70, 50:70] = (255, 255, 255)  # 400 px large spot
    ratio = calculate_largest_glare_region_ratio(image)
    assert ratio == pytest.approx(400 / 10000)


# ---------------------------------------------------------------------------
# Behavior test — glare vs no-glare image
# ---------------------------------------------------------------------------

def test_glare_image_has_higher_area_ratio_than_clean_image(
    colored_image_no_glare, colored_image_with_one_glare_spot
):
    clean_ratio = calculate_glare_area_ratio(colored_image_no_glare)
    glare_ratio = calculate_glare_area_ratio(colored_image_with_one_glare_spot)
    assert glare_ratio > clean_ratio
