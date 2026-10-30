from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffectedArea:
    affected_pixels: int
    leaf_pixels: int
    percentage: float


def calculate_affected_area(leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> AffectedArea:
    if leaf_mask.shape != lesion_mask.shape:
        raise ValueError("Leaf and lesion masks must have the same shape.")

    leaf = np.asarray(leaf_mask, dtype=bool)
    lesion = np.asarray(lesion_mask, dtype=bool) & leaf
    leaf_pixels = int(leaf.sum())
    affected_pixels = int(lesion.sum())
    percentage = 0.0 if leaf_pixels == 0 else (affected_pixels / leaf_pixels) * 100.0
    return AffectedArea(
        affected_pixels=affected_pixels,
        leaf_pixels=leaf_pixels,
        percentage=max(0.0, min(100.0, percentage)),
    )
