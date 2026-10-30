import json

from PIL import Image

from training.evaluate_plantdoc import load_test_records


def test_plantdoc_test_loader_maps_known_labels_and_counts_unmapped_data(tmp_path):
    class_names = ["Tomato___healthy", "Tomato___Early_blight"]
    mapping = {"Tomato leaf": "Tomato___healthy", "Unmapped crop": "not_supported"}
    healthy = tmp_path / "test" / "Tomato leaf" / "healthy.jpg"
    healthy.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(healthy)
    unsupported = tmp_path / "test" / "Unmapped crop" / "unknown.jpg"
    unsupported.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "brown").save(unsupported)
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "split": "test",
                        "class_name": "Tomato leaf",
                        "path": "test/Tomato leaf/healthy.jpg",
                    },
                    {
                        "split": "test",
                        "class_name": "Unmapped crop",
                        "path": "test/Unmapped crop/unknown.jpg",
                    },
                    {
                        "split": "train",
                        "class_name": "Tomato leaf",
                        "path": "train/Tomato leaf/healthy.jpg",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    records, counts = load_test_records(tmp_path, mapping, class_names)

    assert len(records) == 1
    assert records[0]["label"] == "Tomato___healthy"
    assert counts["test_images_in_source_manifest"] == 2
    assert counts["mapped_test_images"] == 1
    assert counts["unmapped_test_images"] == 1
