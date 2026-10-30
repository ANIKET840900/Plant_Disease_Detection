import torch

from training.evaluate_classifier import (
    choose_review_threshold,
    expected_calibration_error,
    fit_temperature,
    metrics_for,
    verifies_training_provenance,
)


def test_temperature_fit_and_review_threshold_use_calibration_scores():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 0])

    temperature = fit_temperature(logits, labels)
    review_policy = choose_review_threshold(torch.softmax(logits / temperature, dim=1), labels)

    assert 0.05 <= temperature <= 20.0
    assert 0.0 <= review_policy["threshold"] <= 1.0
    assert 0.0 <= review_policy["calibration_coverage"] <= 1.0


def test_metrics_report_per_class_and_calibration_error():
    logits = torch.tensor([[8.0, 0.0], [0.0, 8.0], [7.0, 0.0]])
    labels = torch.tensor([0, 1, 0])

    metrics, confusion = metrics_for(logits, labels, ["healthy", "disease"], 1.0, ["a", "b", "c"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["disease"]["support"] == 1
    assert metrics["misclassified"] == []
    assert confusion.tolist() == [[2, 0], [0, 1]]
    assert expected_calibration_error(torch.softmax(logits, dim=1), labels) > 0.0


def test_training_provenance_requires_matching_manifest_and_train_partition():
    metadata = {
        "split_manifest_sha256": "manifest-hash",
        "training_split_name": "train",
        "train_images": 100,
    }

    assert verifies_training_provenance(metadata, "manifest-hash", 100)
    assert not verifies_training_provenance(metadata, "another-hash", 100)
    assert not verifies_training_provenance({}, "manifest-hash", 100)
