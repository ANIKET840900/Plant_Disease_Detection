import json
import threading

import torch
from PIL import Image

from backend.app.config import settings
from backend.app.services.models import ModelRegistry


def _write_checkpoint(directory, labels):
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 2),
    ).eval()
    example = torch.zeros(1, 3, 32, 32)
    torch.jit.trace(model, example).save(str(directory / "classifier.pt"))
    (directory / "class_mapping.json").write_text(json.dumps(labels), encoding="utf-8")
    (directory / "classifier_metadata.json").write_text(
        json.dumps({"classes": labels, "image_size": 32, "calibration_temperature": 1.5}),
        encoding="utf-8",
    )


def test_registry_validates_and_loads_a_compatible_checkpoint(tmp_path, monkeypatch):
    labels = ["Crop___healthy", "Crop___disease"]
    _write_checkpoint(tmp_path, labels)
    monkeypatch.setattr(settings, "model_dir", tmp_path)
    registry = ModelRegistry()

    registry.load()
    result = registry.predict_class(Image.new("RGB", (160, 160), "green"))

    assert registry.health()["classifier_loaded"]
    assert registry.image_size == 32
    assert result["predicted_class"] in labels
    assert result["confidence_calibrated"] is False
    assert result["review_threshold"] == 0.6


def test_registry_rejects_output_width_that_disagrees_with_mapping(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, ["only_one_label"])
    monkeypatch.setattr(settings, "model_dir", tmp_path)
    registry = ModelRegistry()

    registry.load()

    assert not registry.health()["classifier_loaded"]
    assert "output width" in registry.model_load_error


def test_gradcam_returns_attribution_overlay_not_a_lesion_mask():
    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Linear(4, 2)
            with torch.no_grad():
                self.features[0].weight.fill_(0.1)
                self.features[0].bias.fill_(0.1)
                self.classifier.weight.fill_(0.1)
                self.classifier.bias.zero_()

        def forward(self, tensor):
            features = self.features(tensor)
            return self.classifier(torch.flatten(self.pool(features), 1))

    registry = ModelRegistry(
        explainability_model=TinyClassifier().eval(),
        class_names=["Crop___healthy", "Crop___disease"],
        image_size=32,
        explainability_lock=threading.Lock(),
    )

    result = registry.gradcam(Image.new("RGB", (48, 64), "white"), "Crop___disease")

    assert result["image"].startswith("data:image/jpeg;base64,")
    assert result["class_name"] == "Crop___disease"
    assert "not a verified disease lesion mask" in result["explanation"]
