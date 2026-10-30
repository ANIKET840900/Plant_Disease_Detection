import numpy as np

from backend.app.segmentation.measurements import calculate_affected_area


def test_empty_leaf_mask_returns_zero() -> None:
    result = calculate_affected_area(np.zeros((2, 2)), np.ones((2, 2)))
    assert result.leaf_pixels == 0
    assert result.affected_pixels == 0
    assert result.percentage == 0.0


def test_partial_lesion_is_restricted_to_leaf() -> None:
    leaf = np.array([[1, 1], [0, 0]])
    lesion = np.ones((2, 2))
    result = calculate_affected_area(leaf, lesion)
    assert result.leaf_pixels == 2
    assert result.affected_pixels == 2
    assert result.percentage == 100.0


def test_half_leaf_is_affected() -> None:
    leaf = np.ones((2, 2))
    lesion = np.array([[1, 1], [0, 0]])
    result = calculate_affected_area(leaf, lesion)
    assert result.affected_pixels == 2
    assert result.leaf_pixels == 4
    assert result.percentage == 50.0
