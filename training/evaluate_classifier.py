"""Fit temperature on calibration data and report final held-out metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ManifestDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.data_dir = data_dir
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        image_path = (
            self.data_dir
            / Path(record.get("dataset_root", "."))
            / Path(record["path"])
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.class_to_idx[record["label"]], record["path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--promote-if-better", action="store_true")
    return parser.parse_args()


def collect_logits(
    model: torch.jit.ScriptModule,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    logits_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            logits_batches.append(model(images.to(device)).cpu())
            labels_batches.append(labels)
            paths.extend(batch_paths)
    return torch.cat(logits_batches), torch.cat(labels_batches), paths


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    boundaries = torch.linspace(0, 1, bins + 1)
    error = torch.zeros((), dtype=torch.float64)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            error += mask.double().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return float(error)


def metrics_for(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    temperature: float,
    paths: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    predictions = probabilities.argmax(dim=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels.numpy(), predictions.numpy(), labels=list(range(len(class_names))), zero_division=0
    )
    confusion = confusion_matrix(labels.numpy(), predictions.numpy(), labels=list(range(len(class_names))))
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    one_hot = torch.nn.functional.one_hot(labels, num_classes=len(class_names)).float()
    metrics = {
        "examples": len(labels),
        "accuracy": float(accuracy_score(labels.numpy(), predictions.numpy())),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "negative_log_likelihood": float(torch.nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), labels)),
        "brier_score": float(((probabilities - one_hot) ** 2).sum(dim=1).mean()),
        "expected_calibration_error_15_bins": expected_calibration_error(probabilities, labels),
        "per_class": per_class,
        "misclassified": [
            {"path": path, "true_class": class_names[int(truth)], "predicted_class": class_names[int(prediction)]}
            for path, truth, prediction in zip(paths, labels.tolist(), predictions.tolist())
            if truth != prediction
        ],
    }
    return metrics, confusion


def choose_review_threshold(probabilities: torch.Tensor, labels: torch.Tensor, target_accuracy: float = 0.90) -> dict[str, float]:
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    thresholds = torch.unique(confidence).sort().values
    eligible = []
    for threshold in thresholds:
        selected = confidence >= threshold
        coverage = float(selected.float().mean())
        accuracy = float(correct[selected].float().mean()) if selected.any() else 0.0
        if accuracy >= target_accuracy:
            eligible.append((coverage, float(threshold), accuracy))
    if not eligible:
        return {"threshold": 1.0, "target_accuracy": target_accuracy, "calibration_coverage": 0.0}
    coverage, threshold, accuracy = max(eligible)
    return {
        "threshold": threshold,
        "target_accuracy": target_accuracy,
        "calibration_coverage": coverage,
        "calibration_selective_accuracy": accuracy,
    }


def verifies_training_provenance(
    metadata: dict[str, Any],
    manifest_digest: str,
    train_count: int,
) -> bool:
    return (
        metadata.get("split_manifest_sha256") == manifest_digest
        and metadata.get("training_split_name") == "train"
        and metadata.get("train_images") == train_count
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data or Path(config["data_dir"])
    manifest_path = args.manifest or Path(config["split_manifest"])
    model_dir = args.model_dir or Path(config["model_dir"])
    report_dir = args.report_dir or Path(config["report_dir"])
    batch_size = args.batch_size or int(config["batch_size"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    image_folder = datasets.ImageFolder(data_dir)
    class_names = image_folder.classes
    if class_names != manifest["class_names"]:
        raise SystemExit("Dataset classes do not match split manifest.")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    records = manifest["records"]
    train_count = sum(record["split"] == "train" for record in records)
    candidate_metadata_path = model_dir / "classifier_metadata.json"
    candidate_metadata = (
        json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
        if candidate_metadata_path.is_file()
        else {}
    )
    candidate_provenance_verified = verifies_training_provenance(
        candidate_metadata,
        manifest_digest,
        train_count,
    )
    calibration_records = [record for record in records if record["split"] == "calibration"]
    test_records = [record for record in records if record["split"] == "test"]
    if not calibration_records or not test_records:
        raise SystemExit("Manifest requires separate calibration and test records.")
    calibration_loader = DataLoader(
        ManifestDataset(data_dir, calibration_records, image_folder.class_to_idx, transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        ManifestDataset(data_dir, test_records, image_folder.class_to_idx, transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    candidate_path = model_dir / "classifier.pt"
    if not candidate_path.is_file():
        raise SystemExit(f"Candidate checkpoint not found: {candidate_path}")
    candidate = torch.jit.load(str(candidate_path), map_location=device)
    calibration_logits, calibration_labels, _ = collect_logits(candidate, calibration_loader, device)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibrated_probabilities = torch.softmax(calibration_logits / temperature, dim=1)
    review_policy = choose_review_threshold(calibrated_probabilities, calibration_labels)
    test_logits, test_labels, test_paths = collect_logits(candidate, test_loader, device)
    candidate_metrics, candidate_confusion = metrics_for(test_logits, test_labels, class_names, temperature, test_paths)

    baseline_path = args.baseline_dir / "classifier.pt"
    comparison: dict[str, Any] = {}
    baseline_metrics = None
    baseline_provenance_verified = False
    if baseline_path.is_file() and baseline_path.resolve() != candidate_path.resolve():
        baseline_mapping_path = args.baseline_dir / "class_mapping.json"
        baseline_mapping = json.loads(baseline_mapping_path.read_text(encoding="utf-8")) if baseline_mapping_path.exists() else None
        baseline_metadata_path = args.baseline_dir / "classifier_metadata.json"
        baseline_metadata = (
            json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
            if baseline_metadata_path.is_file()
            else {}
        )
        baseline_provenance_verified = verifies_training_provenance(
            baseline_metadata,
            manifest_digest,
            train_count,
        )
        if baseline_mapping != class_names:
            comparison["baseline_test_error"] = (
                "Baseline class_mapping.json is missing or differs from the split class order."
            )
        else:
            baseline = torch.jit.load(str(baseline_path), map_location=device)
            baseline_logits, baseline_labels, baseline_paths = collect_logits(baseline, test_loader, device)
        if (
            baseline_mapping == class_names
            and baseline_logits.shape[1] == len(class_names)
            and torch.equal(baseline_labels, test_labels)
        ):
            baseline_metrics = metrics_for(
                baseline_logits, baseline_labels, class_names, 1.0, baseline_paths
            )[0]
            comparison["baseline_test"] = baseline_metrics
        elif baseline_mapping == class_names:
            comparison["baseline_test_error"] = (
                "Baseline output class count or mapping is incompatible with the current test split."
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    claim_scope = (
        "Candidate training provenance matches this manifest and uses only its train partition. "
        "The test partition is duplicate-group-held-out PlantVillage data, not a plant/session-level "
        "or independent field-photo evaluation."
        if candidate_provenance_verified
        else "The manifest defines a PlantVillage test partition, but this checkpoint's training "
        "provenance is missing or does not match the manifest. Possible train/test overlap cannot "
        "be ruled out; metrics are exploratory only and are not field validation."
    )
    report = {
        "model": str(candidate_path),
        "dataset": str(data_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "claim_scope": claim_scope,
        "evaluation_integrity": {
            "manifest_sha256": manifest_digest,
            "candidate_training_provenance_verified": candidate_provenance_verified,
            "test_independent_of_candidate_training_verified": candidate_provenance_verified,
            "baseline_training_provenance_verified": baseline_provenance_verified,
            "calibration_independent_of_candidate_training_verified": candidate_provenance_verified,
            "plant_or_session_grouping_available": False,
            "field_dataset_evaluated": False,
        },
        "calibration_temperature": temperature,
        "review_policy": review_policy,
        "candidate_test": candidate_metrics,
        **comparison,
    }
    report_path = report_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (report_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["true\\predicted", *class_names])
        for name, row in zip(class_names, candidate_confusion):
            writer.writerow([name, *row.tolist()])
    metadata_path = model_dir / "classifier_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["calibration_temperature"] = temperature
    metadata["review_threshold"] = review_policy["threshold"]
    metadata["review_threshold_calibration"] = review_policy
    metadata["evaluation_report"] = str(report_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    promotion = {"requested": args.promote_if_better, "promoted": False}
    if args.promote_if_better and not candidate_provenance_verified:
        promotion["blocked_reason"] = (
            "Candidate training provenance does not match this split manifest; "
            "its test metrics cannot be treated as independent."
        )
    elif args.promote_if_better and baseline_metrics is not None and not baseline_provenance_verified:
        promotion["blocked_reason"] = (
            "Baseline training provenance does not match this split manifest; "
            "a fair held-out comparison cannot be verified."
        )
    elif args.promote_if_better and baseline_metrics is not None:
        candidate_class_f1 = candidate_metrics["per_class"]
        baseline_class_f1 = baseline_metrics["per_class"]
        severe_class_regression = any(
            baseline_class_f1[name]["f1"] >= 0.5
            and candidate_class_f1[name]["f1"] < baseline_class_f1[name]["f1"] - 0.10
            for name in class_names
        )
        promotion["criteria"] = {
            "macro_f1_improved": candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"],
            "accuracy_not_over_1pct_lower": candidate_metrics["accuracy"] >= baseline_metrics["accuracy"] - 0.01,
            "no_class_with_baseline_f1_0_5_has_over_0_1_drop": not severe_class_regression,
        }
        promotion["promoted"] = all(promotion["criteria"].values())
        if promotion["promoted"]:
            backup = args.baseline_dir / f"baseline_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            backup.mkdir(parents=True, exist_ok=False)
            for filename in ("classifier.pt", "class_mapping.json", "classifier_metadata.json"):
                source = args.baseline_dir / filename
                if source.exists():
                    shutil.copy2(source, backup / filename)
            staging = args.baseline_dir / ".candidate_promotion"
            staging.mkdir(exist_ok=False)
            try:
                for filename in ("classifier.pt", "class_mapping.json", "classifier_metadata.json"):
                    shutil.copy2(model_dir / filename, staging / filename)
                for filename in ("classifier.pt", "class_mapping.json", "classifier_metadata.json"):
                    os.replace(staging / filename, args.baseline_dir / filename)
            finally:
                for staged_file in staging.iterdir():
                    if staged_file.is_file():
                        staged_file.unlink()
                staging.rmdir()
            promotion["baseline_backup"] = str(backup)
            promotion["production_model_dir"] = str(args.baseline_dir)
    report["promotion"] = promotion
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata["promotion"] = promotion
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"temperature={temperature:.4f}")
    print(f"promotion={json.dumps(promotion)}")
    print(json.dumps({key: value for key, value in candidate_metrics.items() if key != "per_class" and key != "misclassified"}, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
