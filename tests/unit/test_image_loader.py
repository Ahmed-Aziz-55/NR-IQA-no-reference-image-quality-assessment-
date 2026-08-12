from pathlib import Path

import cv2
import numpy as np
import pytest

from app.io.image_loader import ImageLoadError, load_image


def test_load_valid_image(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.jpg"

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    loaded = load_image(image_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)
    assert loaded.dtype == np.uint8


def test_load_missing_image() -> None:
    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image("does_not_exist.jpg")


def test_load_directory(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="is not a file"):
        load_image(tmp_path)


def test_load_invalid_image(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.write_text("this is not an image")

    with pytest.raises(ImageLoadError, match="Unable to decode image"):
        load_image(invalid_path)
