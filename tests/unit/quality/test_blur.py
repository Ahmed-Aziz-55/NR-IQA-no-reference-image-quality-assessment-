import cv2
import numpy as np
import pytest

from app.quality.classical.blur import calculate_laplacian_variance


def test_calculate_laplacian_variance_bgr_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)

    score = calculate_laplacian_variance(image)

    assert isinstance(score, float)
    assert score > 0.0


def test_calculate_laplacian_variance_grayscale_image() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)

    score = calculate_laplacian_variance(image)

    assert isinstance(score, float)
    assert score > 0.0


def test_calculate_laplacian_variance_bgra_image() -> None:
    image = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.rectangle(
        image,
        (20, 20),
        (80, 80),
        (255, 255, 255, 255),
        -1,
    )

    score = calculate_laplacian_variance(image)

    assert isinstance(score, float)
    assert score > 0.0


def test_empty_image_raises_value_error() -> None:
    image = np.empty((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Image cannot be empty"):
        calculate_laplacian_variance(image)


def test_non_numpy_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Expected a NumPy array"):
        calculate_laplacian_variance([[1, 2], [3, 4]])


def test_unsupported_channel_count_raises_value_error() -> None:
    image = np.zeros((100, 100, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported channel count"):
        calculate_laplacian_variance(image)


def test_too_small_image_raises_value_error() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Image too small"):
        calculate_laplacian_variance(image)


@pytest.mark.parametrize("ksize", [0, 2, 4, 6])
def test_invalid_ksize_raises_value_error(ksize: int) -> None:
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="ksize must be a positive odd integer",
    ):
        calculate_laplacian_variance(image, ksize=ksize)


def test_valid_ksize_is_accepted() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)

    score = calculate_laplacian_variance(image, ksize=3)

    assert isinstance(score, float)
    assert score > 0.0


def test_blurring_image_reduces_laplacian_variance() -> None:
    sharp_image = np.zeros((200, 200), dtype=np.uint8)

    cv2.rectangle(
        sharp_image,
        (40, 40),
        (160, 160),
        255,
        -1,
    )

    blurred_image = cv2.GaussianBlur(
        sharp_image,
        (21, 21),
        0,
    )

    sharp_score = calculate_laplacian_variance(sharp_image)
    blurred_score = calculate_laplacian_variance(blurred_image)

    assert sharp_score > blurred_score
