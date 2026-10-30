from PIL import Image
import pytest

from backend.app.services.screening import _windows, screen_plant_image


class StubRegistry:
    classifier = object()
    class_names = ["Tomato___healthy", "Tomato___Early_blight"]

    def __init__(self, predicted_class: str = "Tomato___Early_blight", confidence: float = 0.91) -> None:
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.calls = 0

    def predict_class(self, image: Image.Image) -> dict[str, object]:
        self.calls += 1
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_scores": {self.predicted_class: self.confidence},
        }


def test_single_leaf_returns_candidate_crops_and_original_overlay() -> None:
    registry = StubRegistry()
    result = screen_plant_image(Image.new("RGB", (256, 256), "green"), registry, "single_leaf")

    assert result["mode"] == "single_leaf"
    assert result["overall_prediction"]["predicted_class"] == "Tomato___Early_blight"
    assert result["regions_checked"] > 1
    assert result["candidate_count"] > 0
    assert result["overlay"].startswith("data:image/jpeg;base64,")
    assert all(region["image"].startswith("data:image/jpeg;base64,") for region in result["candidate_regions"])
    assert all(region["severity"] == "Not assessed from image" for region in result["candidate_regions"])


def test_healthy_predictions_are_not_shown_as_affected_candidates() -> None:
    result = screen_plant_image(
        Image.new("RGB", (320, 320), "green"),
        StubRegistry("Tomato___healthy", 0.94),
        "whole_plant",
    )

    assert result["candidate_count"] == 0
    assert result["candidate_regions"] == []
    assert "No candidates does not prove" in result["limitations"][2]


def test_low_score_is_not_reported_as_candidate() -> None:
    result = screen_plant_image(
        Image.new("RGB", (320, 320), "green"),
        StubRegistry("Tomato___Early_blight", 0.59),
        "whole_plant",
    )

    assert result["candidate_count"] == 0
    assert result["candidate_score_threshold"] == 0.60


def test_calibrated_scores_and_review_threshold_are_reported_without_claiming_ood() -> None:
    registry = StubRegistry(confidence=0.91)
    registry.calibration_report = "held-out calibration report"
    registry.review_threshold = 0.95

    result = screen_plant_image(Image.new("RGB", (256, 256), "green"), registry, "single_leaf")

    assert result["confidence_calibrated"] is True
    assert result["review_required"] is True
    assert result["unknown_detection_validated"] is False
    assert result["candidate_regions"][0]["confidence_calibrated"] is True
    assert result["candidate_regions"][0]["needs_review"] is True
    assert "not independent botanical" in result["candidate_regions"][0]["explanation"]
    assert "temperature-scaled" in result["candidate_regions"][0]["visible_evidence"]


def test_invalid_screening_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Mode must be"):
        screen_plant_image(Image.new("RGB", (256, 256)), StubRegistry(), "unsupported")


def test_wide_plant_screen_samples_across_full_image() -> None:
    boxes = _windows(Image.new("RGB", (4096, 256)), "whole_plant")

    assert len(boxes) <= 36
    assert min(box[0] for box in boxes) == 0
    assert max(box[2] for box in boxes) == 4096
