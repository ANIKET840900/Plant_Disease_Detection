"""Reproducibly train a MobileNetV3-Small candidate on a prepared split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as transform_functional

ImageFile.LOAD_TRUNCATED_IMAGES = False


class RandomMotionBlur:
    """Apply a short directional blur to simulate camera or subject motion."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        direction = random.randrange(4)
        kernel = torch.zeros((1, 1, 5, 5))
        for index in range(5):
            if direction == 0:
                row, column = 2, index
            elif direction == 1:
                row, column = index, 2
            elif direction == 2:
                row, column = index, index
            else:
                row, column = index, 4 - index
            kernel[0, 0, row, column] = 0.2
        tensor = transform_functional.to_tensor(image.convert("RGB")).unsqueeze(0)
        channels = tensor.shape[1]
        blurred = torch.nn.functional.conv2d(
            tensor,
            kernel.expand(channels, 1, 5, 5),
            padding=2,
            groups=channels,
        )
        return transform_functional.to_pil_image(blurred.squeeze(0).clamp(0, 1))


class RandomShadow:
    """Simulate a soft cast shadow over a random portion of the image."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        width, height = image.size
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        points = [
            (random.randrange(width), random.randrange(height))
            for _ in range(4)
        ]
        points.extend(
            [
                (random.randrange(width), random.randrange(height))
                for _ in range(2)
            ]
        )
        draw.polygon(points, fill=random.randint(90, 180))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, min(image.size) // 12)))
        darkened = ImageEnhance.Brightness(image.convert("RGB")).enhance(
            random.uniform(0.45, 0.8)
        )
        return Image.composite(darkened, image.convert("RGB"), mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def build_model(class_count: int, head_dropout: float) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError("head_dropout must be in the range [0, 1).")
    model.classifier[-2] = nn.Dropout(p=head_dropout, inplace=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = (
            self.data_dir
            / Path(record.get("dataset_root", "."))
            / Path(record["path"])
        )
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.class_to_idx[record["label"]]


def evaluate_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    count = 0
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            count += labels.size(0)
            encoded = labels.cpu() * class_count + predictions.cpu()
            confusion += torch.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    present = support > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    accuracy = float(true_positive.sum() / max(count, 1))
    if count == 0:
        raise RuntimeError("Validation loader returned no examples.")
    if not np.isfinite(macro_f1):
        raise RuntimeError("Validation macro-F1 was non-finite.")
    return loss_sum / count, macro_f1


def set_frozen_stages(model: nn.Module, train_backbone: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = train_backbone
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def build_fine_tune_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Apply linear warmup followed by cosine decay over fine-tuning steps."""
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(total_steps, steps_per_epoch * warmup_epochs)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data or Path(config["data_dir"])
    manifest_path = args.manifest or Path(config["split_manifest"])
    model_dir = args.model_dir or Path(config["model_dir"])
    epochs = args.epochs or int(config["epochs"])
    batch_size = args.batch_size or int(config["batch_size"])
    seed = int(config["seed"])

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset not found: {data_dir}. Run training/prepare_splits.py first.")
    if not manifest_path.is_file():
        raise SystemExit(f"Split manifest not found: {manifest_path}. Run training/prepare_splits.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_everything(seed, bool(config.get("deterministic", True)))

    eval_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    augmentation = config["augmentation"]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config["image_size"],
            scale=tuple(augmentation["crop_scale"]),
            ratio=tuple(augmentation["crop_ratio"]),
        ),
        transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip_probability"]),
        transforms.RandomVerticalFlip(p=augmentation["vertical_flip_probability"]),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=augmentation["brightness"],
                    contrast=augmentation["contrast"],
                    saturation=augmentation["saturation"],
                    hue=augmentation["hue"],
                )
            ],
            p=augmentation["color_jitter_probability"],
        ),
        transforms.RandomAffine(
            degrees=augmentation["rotation_degrees"],
            translate=(augmentation["translation_fraction"],) * 2,
            scale=tuple(augmentation["scale_range"]),
            shear=augmentation["shear_degrees"],
        ),
        transforms.RandomPerspective(
            distortion_scale=augmentation["perspective_distortion"],
            p=augmentation["perspective_probability"],
        ),
        RandomMotionBlur(augmentation["motion_blur_probability"]),
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=tuple(augmentation["gaussian_blur_sigma"]),
                )
            ],
            p=augmentation["gaussian_blur_probability"],
        ),
        RandomShadow(augmentation["shadow_probability"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=augmentation["random_erasing_probability"],
            scale=tuple(augmentation["random_erasing_scale"]),
            ratio=tuple(augmentation["random_erasing_ratio"]),
        ),
    ])

    image_folder = datasets.ImageFolder(data_dir)
    class_names = image_folder.classes
    if class_names != manifest["class_names"]:
        raise SystemExit("Dataset class directories do not match split manifest classes; regenerate the manifest.")
    records = manifest["records"]
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    if not train_records or not validation_records:
        raise SystemExit("Split manifest must include training and validation records.")
    training_labels = {record["label"] for record in train_records}
    missing_training_labels = sorted(set(class_names) - training_labels)
    if missing_training_labels:
        raise SystemExit(f"Some configured classes have no training examples after deduplication: {missing_training_labels}")

    train_dataset = ManifestDataset(data_dir, train_records, image_folder.class_to_idx, train_transform)
    validation_dataset = ManifestDataset(data_dir, validation_records, image_folder.class_to_idx, eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = max(1, min(torch.get_num_threads(), 6))
    torch.set_num_threads(threads)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        len(class_names),
        float(config["head_dropout"]),
    ).to(device)
    class_counts = torch.bincount(
        torch.tensor([image_folder.class_to_idx[record["label"]] for record in train_records]),
        minlength=len(class_names),
    ).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1)).sqrt()
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=float(config["label_smoothing"]),
    )
    best_validation_loss = float("inf")
    best_validation_macro_f1 = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    fine_tune_optimizer: torch.optim.Optimizer | None = None
    fine_tune_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    log_path = model_dir / "training_log.jsonl"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, model_dir / "training_config.yaml")
    if log_path.exists():
        log_path.unlink()
    print(
        f"device={device} cpu_threads={threads} train={len(train_dataset)} "
        f"validation={len(validation_dataset)} classes={len(class_names)} "
        f"batch_size={batch_size} epochs={epochs}",
        flush=True,
    )

    warmup_epochs = int(config["head_warmup_epochs"])
    if warmup_epochs < 1 or warmup_epochs >= epochs:
        raise SystemExit("head_warmup_epochs must be at least one and less than epochs.")

    for epoch in range(1, epochs + 1):
        fine_tune = epoch > warmup_epochs
        set_frozen_stages(model, fine_tune)
        if fine_tune and fine_tune_optimizer is None:
            fine_tune_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.features.parameters(),
                        "lr": float(config["backbone_learning_rate"]),
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": float(config["head_fine_tune_learning_rate"]),
                    },
                ],
                weight_decay=float(config["weight_decay"]),
            )
            fine_tune_scheduler = build_fine_tune_scheduler(
                fine_tune_optimizer,
                len(train_loader),
                epochs - warmup_epochs,
                int(config["cosine_warmup_epochs"]),
            )
        if fine_tune:
            optimizer = fine_tune_optimizer
            scheduler = fine_tune_scheduler
        else:
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            scheduler = None

        model.train()
        if not fine_tune:
            model.features.eval()
        running_loss = 0.0
        epoch_started = time.monotonic()
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item() * labels.size(0)
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        validation_loss, validation_macro_f1 = evaluate_macro_f1(
            model, validation_loader, device, len(class_names)
        )
        train_loss = running_loss / len(train_dataset)
        epoch_seconds = time.monotonic() - epoch_started
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_macro_f1,
            "learning_rates": current_lrs,
            "fine_tuned_backbone": fine_tune,
            "seconds": epoch_seconds,
        }
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(result) + "\n")
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={validation_macro_f1:.4f} "
            f"learning_rates={current_lrs} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        minimum_delta = float(config["minimum_validation_loss_delta"])
        if validation_loss < best_validation_loss - minimum_delta:
            best_validation_loss = validation_loss
            best_validation_macro_f1 = validation_macro_f1
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, model_dir / "best_state.pt")
        else:
            stale_epochs += 1
            if fine_tune and stale_epochs >= int(config["patience"]):
                print(
                    f"early_stopping=validation_loss patience={config['patience']} "
                    f"best_validation_loss={best_validation_loss:.4f}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint candidate.")
    model.load_state_dict(best_state)
    model.eval().cpu()
    torch.jit.script(model).save(str(model_dir / "classifier.pt"))
    (model_dir / "class_mapping.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metadata = {
        "architecture": "MobileNetV3-Small",
        "head_dropout": float(config["head_dropout"]),
        "classes": class_names,
        "image_size": config["image_size"],
        "seed": seed,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "best_validation_loss": best_validation_loss,
        "best_validation_macro_f1": best_validation_macro_f1,
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_split_name": "train",
        "calibration_temperature": 1.0,
        "training_config": config,
    }
    (model_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"saved={model_dir / 'classifier.pt'} "
        f"best_validation_loss={best_validation_loss:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
