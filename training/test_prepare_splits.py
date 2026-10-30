import json

import numpy as np
from PIL import Image

from training.prepare_splits import build_manifest


def _write_image(path, seed):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_manifest_is_reproducible_stratified_and_excludes_cross_label_duplicates(tmp_path):
    data_dir = tmp_path / "dataset"
    for class_index, label in enumerate(("crop_a_disease", "crop_b_healthy")):
        class_dir = data_dir / label
        class_dir.mkdir(parents=True)
        for image_index in range(12):
            _write_image(class_dir / f"image-{image_index}.png", class_index * 100 + image_index)

    duplicate_source = data_dir / "crop_a_disease" / "image-0.png"
    (data_dir / "crop_b_healthy" / "mislabelled-duplicate.png").write_bytes(duplicate_source.read_bytes())
    (data_dir / "crop_a_disease" / "redundant-copy.png").write_bytes(
        (data_dir / "crop_a_disease" / "image-1.png").read_bytes()
    )
    (data_dir / "crop_a_disease" / "broken.png").write_bytes(b"not an image")

    fractions = {"train": 0.5, "validation": 0.2, "calibration": 0.15, "test": 0.15}
    first = build_manifest(data_dir, fractions, seed=7, near_distance=3)
    second = build_manifest(data_dir, fractions, seed=7, near_distance=3)

    assert first == second
    assert len(first["rejected_corrupt_images"]) == 1
    assert len(first["excluded_cross_label_duplicate_images"]) == 2
    assert first["removed_exact_duplicate_images"] == [
        {"path": "crop_a_disease/redundant-copy.png", "label": "crop_a_disease"}
    ]
    assert all(record["path"] != "crop_a_disease/redundant-copy.png" for record in first["records"])
    assert all(record["label"] != "crop_b_healthy" or record["path"] != "crop_b_healthy/mislabelled-duplicate.png"
               for record in first["records"])
    for label in first["class_names"]:
        represented_splits = {
            record["split"] for record in first["records"] if record["label"] == label
        }
        assert represented_splits == {"train", "validation", "calibration", "test"}
    splits_by_group = {}
    for record in first["records"]:
        splits_by_group.setdefault(record["duplicate_group"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    json.dumps(first)
