from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    format: str
    blur_score: float
    is_blurry: bool


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded or is too small to analyze."""


def decode_image(contents: bytes, minimum_dimension: int = 128) -> tuple[Image.Image, ImageQuality]:
    if not contents:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(contents)) as source:
            original_format = source.format or "unknown"
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc

    width, height = image.size
    if min(width, height) < minimum_dimension:
        raise InvalidImageError(
            f"Image dimensions must be at least {minimum_dimension}x{minimum_dimension} pixels."
        )

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return image, ImageQuality(
        width=width,
        height=height,
        format=original_format,
        blur_score=blur_score,
        is_blurry=blur_score < 25.0,
    )
