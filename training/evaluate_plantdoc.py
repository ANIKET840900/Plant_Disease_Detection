"""Evaluate a trained classifier on PlantDoc's official held-out test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/PlantDoc_dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("saved_models"))
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("training/plantdoc_label_mapping.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/generated/plantdoc_evaluation.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
) -> float:
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    boundaries = torch.linspace(0, 1, bins + 1)
    error = torch.zeros((), dtype=torch.float64)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            error += mask.double().mean() * (
                confidence[mask].mean() - correct[mask].float().mean()
            ).abs()
    return float(error)


def load_test_records(
    data_dir: Path,
    mapping: dict[str, str],
    class_names: list[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    manifest_path = data_dir / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PlantDoc source manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {
        "test_images_in_source_manifest": 0,
        "mapped_test_images": 0,
        "unmapped_test_images": 0,
        "missing_or_corrupt_test_images": 0,
    }
    records: list[dict[str, str]] = []
    root = data_dir.resolve()
    for item in manifest.get("records", []):
        if item.get("split") != "test":
            continue
        counts["test_images_in_source_manifest"] += 1
        source_label = item.get("class_name")
        target_label = mapping.get(source_label)
        if target_label not in class_names:
            counts["unmapped_test_images"] += 1
            continue
        path = (data_dir / item["path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"PlantDoc test path escapes the dataset directory: {item['path']}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError):
            counts["missing_or_corrupt_test_images"] += 1
            continue
        records.append({"path": str(path), "label": target_label})
        counts["mapped_test_images"] += 1
    if not records:
        raise ValueError("No valid PlantDoc test images have supported class mappings.")
    return records, counts


def evaluate(
    data_dir: Path,
    model_dir: Path,
    mapping_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_names = json.loads((model_dir / "class_mapping.json").read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not isinstance(class_names, list):
        raise ValueError("Class mappings must be a JSON object and model labels a JSON list.")
    records, counts = load_test_records(data_dir, mapping, class_names)
    metadata = json.loads(
        (model_dir / "classifier_metadata.json").read_text(encoding="utf-8")
    )
    image_size = int(metadata.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    classifier = torch.jit.load(str(model_dir / "classifier.pt"), map_location="cpu")
    classifier.eval()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    all_logits: list[torch.Tensor] = []
    all_labels: list[int] = []
    batch_size = max(1, batch_size)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = []
            labels = []
            for record in batch_records:
                with Image.open(record["path"]) as source:
                    images.append(transform(source.convert("RGB")))
                labels.append(class_to_index[record["label"]])
            logits = classifier(torch.stack(images))
            if logits.ndim != 2 or logits.shape[1] != len(class_names):
                raise ValueError("Classifier output width does not match the saved class mapping.")
            if not torch.isfinite(logits).all():
                raise ValueError("Classifier returned non-finite PlantDoc scores.")
            all_logits.append(logits.cpu())
            all_labels.extend(labels)

    logits = torch.cat(all_logits)
    labels = torch.tensor(all_labels, dtype=torch.long)
    temperature = float(metadata.get("calibration_temperature", 1.0))
    probabilities = torch.softmax(logits / temperature, dim=1)
    predictions = probabilities.argmax(dim=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels.numpy(),
        predictions.numpy(),
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels.numpy(),
        predictions.numpy(),
        labels=list(range(len(class_names))),
    )
    evaluated_classes = sorted(set(all_labels))
    per_class = {
        class_names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in evaluated_classes
    }
    return {
        "dataset": "PlantDoc official test split",
        "dataset_source": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "source_license": "CC BY 4.0; retain attribution from the dataset README and LICENSE.txt.",
        "claim_scope": (
            "External natural-background dataset evaluation. Folder labels were mapped to "
            "PlantVillage model classes; ambiguous or unsupported labels are excluded and counted. "
            "This is not a field-session-level split, disease confirmation, or OOD validation."
        ),
        "model_dir": str(model_dir.resolve()),
        "image_size": image_size,
        "calibration_temperature_applied": temperature,
        "calibration_provenance_note": (
            "The model checkpoint temperature is used as stored. If calibrated only on PlantVillage, "
            "ECE on PlantDoc is an external-domain calibration measurement, not recalibration."
        ),
        "counts": counts,
        "examples": len(labels),
        "accuracy": float((predictions == labels).float().mean()),
        "macro_precision": float(np.mean([precision[index] for index in evaluated_classes])),
        "macro_recall": float(np.mean([recall[index] for index in evaluated_classes])),
        "macro_f1": float(np.mean([f1[index] for index in evaluated_classes])),
        "expected_calibration_error_15_bins": expected_calibration_error(
            probabilities,
            labels,
        ),
        "evaluated_class_count": len(evaluated_classes),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [class_names[index] for index in evaluated_classes],
            "values": matrix[np.ix_(evaluated_classes, evaluated_classes)].tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args.data, args.model_dir, args.mapping, args.batch_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"per_class", "confusion_matrix"}
    }
    print(json.dumps(summary, indent=2))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
