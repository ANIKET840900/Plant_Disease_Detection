import base64
import threading
import json
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import models as torchvision_models
from torchvision import transforms

from ..config import settings


LockType = type(threading.Lock())


class ModelUnavailableError(RuntimeError):
    """Raised when a trained checkpoint is not available for inference."""


@dataclass
class ModelRegistry:
    classifier: Any = None
    explainability_model: Any = None
    segmentation: Any = None
    class_names: list[str] | None = None
    image_size: int = 224
    temperature: float = 1.0
    review_threshold: float = 0.60
    model_load_error: str | None = None
    explainability_error: str | None = None
    segmentation_load_error: str | None = None
    calibration_report: str | None = None
    explainability_lock: LockType | None = None

    def load(self) -> None:
        self.classifier = None
        self.explainability_model = None
        self.segmentation = None
        self.class_names = None
        self.model_load_error = None
        self.explainability_error = None
        self.segmentation_load_error = None
        self.explainability_lock = threading.Lock()
        mapping_path = settings.model_dir / "class_mapping.json"
        classifier_path = settings.model_dir / "classifier.pt"
        segmentation_path = settings.model_dir / "segmentation.pt"
        metadata_path = settings.model_dir / "classifier_metadata.json"
        try:
            if mapping_path.exists():
                class_names = json.loads(mapping_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(class_names, list)
                    or not class_names
                    or any(not isinstance(name, str) or not name.strip() for name in class_names)
                    or len(set(class_names)) != len(class_names)
                ):
                    raise ValueError("class_mapping.json must contain unique, non-empty class names.")
            else:
                class_names = None

            if classifier_path.exists():
                if class_names is None:
                    raise ValueError("classifier.pt exists, but class_mapping.json is missing or empty.")
                classifier = torch.jit.load(str(classifier_path), map_location="cpu")
                classifier.eval()
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata_classes = metadata.get("classes")
                    if metadata_classes is not None and metadata_classes != class_names:
                        raise ValueError("Classifier metadata classes do not match class_mapping.json.")
                image_size = int(metadata.get("image_size", 224))
                temperature = float(metadata.get("calibration_temperature", 1.0))
                review_threshold = float(metadata.get("review_threshold", 0.60))
                if image_size < 32 or not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("Classifier metadata has invalid image_size or calibration_temperature.")
                if not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
                    raise ValueError("Classifier metadata review_threshold must be between zero and one.")
                with torch.inference_mode():
                    output = classifier(torch.zeros(1, 3, image_size, image_size))
                if (
                    not isinstance(output, torch.Tensor)
                    or output.ndim != 2
                    or output.shape[0] != 1
                    or output.shape[1] != len(class_names)
                ):
                    output_width = output.shape[-1] if isinstance(output, torch.Tensor) and output.ndim else "unknown"
                    raise ValueError(
                        f"Classifier output width ({output_width}) does not match "
                        f"{len(class_names)} class labels."
                    )
                self.classifier = classifier
                self.class_names = class_names
                self.image_size = image_size
                self.temperature = temperature
                self.review_threshold = review_threshold
                self.calibration_report = metadata.get("evaluation_report")
                if metadata.get("architecture") == "MobileNetV3-Small":
                    try:
                        explainability_model = torchvision_models.mobilenet_v3_small(weights=None)
                        explainability_model.classifier[-1] = torch.nn.Linear(
                            explainability_model.classifier[-1].in_features,
                            len(class_names),
                        )
                        explainability_model.load_state_dict(classifier.state_dict())
                        explainability_model.eval()
                        self.explainability_model = explainability_model
                    except (RuntimeError, ValueError) as exc:
                        self.explainability_error = str(exc)
                else:
                    self.explainability_error = (
                        "Grad-CAM is currently available only for MobileNetV3-Small checkpoints."
                    )
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.model_load_error = str(exc)
            self.classifier = None
            self.explainability_model = None
            self.class_names = None

        if segmentation_path.exists():
            try:
                segmentation = torch.jit.load(str(segmentation_path), map_location="cpu")
                segmentation.eval()
                self.segmentation = segmentation
            except (OSError, RuntimeError) as exc:
                self.segmentation_load_error = str(exc)

    def health(self) -> dict[str, bool]:
        return {
            "classifier_loaded": self.classifier is not None and bool(self.class_names),
            "segmentation_loaded": self.segmentation is not None,
        }

    def predict_class(self, image: Image.Image) -> dict[str, Any]:
        if self.classifier is None or not self.class_names:
            raise ModelUnavailableError(
                "A trained classifier checkpoint and class_mapping.json are required."
            )

        tensor = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = self.classifier(tensor)
            if logits.ndim != 2 or logits.shape != (1, len(self.class_names)):
                raise ModelUnavailableError("The classifier output no longer matches its saved class mapping.")
            if not torch.isfinite(logits).all():
                raise ModelUnavailableError("The classifier returned non-finite scores.")
            probabilities = torch.softmax(logits / self.temperature, dim=1)[0]
        index = int(probabilities.argmax())
        scores = {
            name: float(probabilities[i])
            for i, name in enumerate(self.class_names)
            if i < len(probabilities)
        }
        return {
            "predicted_class": self.class_names[index],
            "class_scores": scores,
            "confidence": float(probabilities[index]),
            "needs_review": float(probabilities[index]) < self.review_threshold,
            "review_threshold": self.review_threshold,
            "confidence_calibrated": self.calibration_report is not None,
        }

    def gradcam(self, image: Image.Image, class_name: str | None = None) -> dict[str, str]:
        """Create a Grad-CAM attribution overlay, not a lesion segmentation mask."""
        if self.explainability_model is None or not self.class_names:
            raise ModelUnavailableError(
                self.explainability_error or "Grad-CAM is unavailable for this checkpoint."
            )
        if self.explainability_lock is None:
            raise ModelUnavailableError("The Grad-CAM model has not completed initialization.")
        if class_name is not None and class_name not in self.class_names:
            raise ValueError(f"Unknown Grad-CAM target class: {class_name}")

        model = self.explainability_model
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        captured: dict[str, torch.Tensor] = {}

        def capture_activations(_: Any, __: Any, output: torch.Tensor) -> None:
            captured["activations"] = output
            output.register_hook(lambda gradient: captured.__setitem__("gradients", gradient))

        with self.explainability_lock:
            handle = model.features[-1].register_forward_hook(capture_activations)
            try:
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = model(tensor)
                    target_index = (
                        self.class_names.index(class_name)
                        if class_name is not None
                        else int(logits.argmax(dim=1).item())
                    )
                    logits[0, target_index].backward()
                activations = captured["activations"].detach()
                gradients = captured["gradients"].detach()
            finally:
                handle.remove()
                model.zero_grad(set_to_none=True)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        maximum = float(cam.max())
        if not math.isfinite(maximum) or maximum <= 0:
            raise ModelUnavailableError("Grad-CAM produced an empty or invalid attribution map.")
        cam = (cam / maximum).cpu().numpy()

        heatmap = np.stack(
            (
                cam,
                np.clip(1.0 - np.abs(cam - 0.5) * 2.0, 0.0, 1.0),
                1.0 - cam,
            ),
            axis=-1,
        )
        heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="RGB")
        overlay = Image.blend(image.convert("RGB"), heatmap_image, 0.35)
        buffer = BytesIO()
        overlay.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "image": f"data:image/jpeg;base64,{encoded}",
            "class_name": self.class_names[target_index],
            "explanation": (
                "Grad-CAM highlights image features influencing this class score. "
                "It is not a verified disease lesion mask or affected-area measurement."
            ),
        }

    def segment(self, image: Image.Image) -> tuple[Any, Any]:
        if self.segmentation is None:
            raise ModelUnavailableError("A trained segmentation checkpoint is required.")

        tensor = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            output = self.segmentation(tensor)
        if output.shape[1] < 2:
            raise RuntimeError("The segmentation checkpoint must output leaf and lesion channels.")
        return output[0, 0].sigmoid() > 0.5, output[0, 1].sigmoid() > 0.5


model_registry = ModelRegistry()
