import torch
from torch import nn
from PIL import Image

from train_classifier import (
    RandomMotionBlur,
    RandomShadow,
    build_fine_tune_scheduler,
    build_model,
    set_frozen_stages,
)


def test_backbone_is_frozen_for_head_warmup_and_fully_unfrozen_for_finetuning():
    model = nn.Module()
    model.features = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model.classifier = nn.Linear(4, 2)

    set_frozen_stages(model, train_backbone=False)
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    set_frozen_stages(model, train_backbone=True)
    assert all(parameter.requires_grad for parameter in model.features.parameters())


def test_finetune_scheduler_warms_up_then_cosine_decays():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_fine_tune_scheduler(
        optimizer,
        steps_per_epoch=2,
        epochs=3,
        warmup_epochs=1,
    )

    assert optimizer.param_groups[0]["lr"] == 0.5
    for _ in range(6):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] < 0.01


def test_field_sensor_augmentations_preserve_rgb_image_shape():
    image = Image.new("RGB", (48, 32), "green")

    assert RandomMotionBlur(1.0)(image).size == image.size
    assert RandomShadow(1.0)(image).size == image.size


def test_model_uses_the_configured_dropout_and_class_projection():
    model = build_model(class_count=3, head_dropout=0.3)

    assert model.classifier[-2].p == 0.3
    assert model.classifier[-1].out_features == 3
