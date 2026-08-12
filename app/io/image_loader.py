from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a BGR NumPy array.

    Raises:
        ImageLoadError: If the path does not exist, is not a file,
            or OpenCV cannot decode the image.
    """
    path = Path(image_path)

    if not path.exists():
        raise ImageLoadError(f"Image path does not exist: {path}")

    if not path.is_file():
        raise ImageLoadError(f"Image path is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError(f"Unable to decode image: {path}")

    return image
