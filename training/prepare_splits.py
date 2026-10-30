"""Validate an ImageFolder dataset and create reproducible duplicate-aware splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "validation", "calibration", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/PlantVillage"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    # config.yaml is JSON, which is also a valid YAML 1.2 document; parsing it
    # with the standard library avoids an extra runtime dependency.
    return json.loads(path.read_text(encoding="utf-8"))


def image_hashes(path: Path) -> tuple[str, int]:
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    with Image.open(io.BytesIO(contents)) as image:
        image.load()
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    perceptual_hash = 0
    for row in range(8):
        for column in range(8):
            perceptual_hash = (perceptual_hash << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return digest, perceptual_hash


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def duplicate_groups(
    records: list[dict[str, Any]],
    near_distance: int,
) -> tuple[list[int], set[int], set[int]]:
    union_find = UnionFind(len(records))
    exact_hashes: dict[str, int] = {}
    redundant_exact_duplicates: set[int] = set()
    hash_buckets: dict[tuple[str, int], list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        digest = record["sha256"]
        if digest in exact_hashes:
            previous = exact_hashes[digest]
            if records[previous]["label"] == record["label"]:
                redundant_exact_duplicates.add(index)
            else:
                union_find.union(index, previous)
        else:
            exact_hashes[digest] = index

        # Compare perceptual hashes only within the same class. Exact matches
        # are still grouped across labels and excluded if labels conflict.
        value = record["difference_hash"]
        for chunk in range(4):
            bucket_value = (value >> (chunk * 16)) & 0xFFFF
            bucket_key = (record["label"], (chunk << 16) | bucket_value)
            for candidate in hash_buckets[bucket_key]:
                if hamming_distance(value, records[candidate]["difference_hash"]) <= near_distance:
                    union_find.union(index, candidate)
            hash_buckets[bucket_key].append(index)

    roots: dict[int, int] = {}
    group_ids: list[int] = []
    labels_by_group: dict[int, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        root = union_find.find(index)
        if root not in roots:
            roots[root] = len(roots)
        group_id = roots[root]
        group_ids.append(group_id)
        labels_by_group[group_id].add(record["label"])
    conflicting_groups = {
        group_id for group_id, labels in labels_by_group.items() if len(labels) > 1
    }
    return group_ids, conflicting_groups, redundant_exact_duplicates


def assign_grouped_splits(
    records: list[dict[str, Any]],
    group_ids: list[int],
    fractions: dict[str, float],
    seed: int,
) -> None:
    if set(fractions) != set(SPLIT_NAMES) or abs(sum(fractions.values()) - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must define {SPLIT_NAMES} and sum to 1.")

    by_label_group: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        by_label_group[record["label"]][group_ids[index]].append(index)

    rng = random.Random(seed)
    for label, groups in by_label_group.items():
        group_items = list(groups.items())
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        target_counts = {split: sum(len(indices) for _, indices in group_items) * fractions[split] for split in SPLIT_NAMES}
        assigned_counts = {split: 0 for split in SPLIT_NAMES}

        for _, indices in group_items:
            split = min(
                SPLIT_NAMES,
                key=lambda name: (
                    assigned_counts[name] - target_counts[name],
                    assigned_counts[name] / max(target_counts[name], 1),
                    SPLIT_NAMES.index(name),
                ),
            )
            assigned_counts[split] += len(indices)
            for index in indices:
                records[index]["split"] = split


def build_manifest(
    data_dir: Path,
    fractions: dict[str, float],
    seed: int,
    near_distance: int,
) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two labeled class directories in {data_dir}.")

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    image_paths_by_class = {
        class_dir: sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for class_dir in class_dirs
    }
    total_paths = sum(len(paths) for paths in image_paths_by_class.values())
    scanned = 0
    for class_dir in class_dirs:
        image_paths = image_paths_by_class[class_dir]
        for path in image_paths:
            scanned += 1
            try:
                digest, perceptual_hash = image_hashes(path)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as error:
                rejected.append({
                    "path": path.relative_to(data_dir).as_posix(),
                    "reason": type(error).__name__,
                })
                continue
            records.append({
                "path": path.relative_to(data_dir).as_posix(),
                "label": class_dir.name,
                "sha256": digest,
                "difference_hash": f"{perceptual_hash:016x}",
            })
            if scanned % 1000 == 0:
                print(f"validated={scanned}/{total_paths}", flush=True)

    labels = sorted({record["label"] for record in records})
    if len(labels) < 2:
        raise ValueError("Fewer than two classes contain valid images after validation.")

    numeric_hash_records = [
        {**record, "difference_hash": int(record["difference_hash"], 16)}
        for record in records
    ]
    if near_distance < 0 or near_distance > 3:
        raise ValueError("Near-duplicate Hamming distance must be between zero and three.")
    group_ids, conflicting_groups, redundant_exact_duplicates = duplicate_groups(numeric_hash_records, near_distance)
    exact_duplicate_records = [
        {"path": record["path"], "label": record["label"]}
        for index, record in enumerate(records)
        if index in redundant_exact_duplicates and group_ids[index] not in conflicting_groups
    ]
    excluded = [
        record for record, group_id in zip(records, group_ids)
        if group_id in conflicting_groups
    ]
    eligible = [
        (record, group_id) for index, (record, group_id) in enumerate(zip(records, group_ids))
        if group_id not in conflicting_groups and index not in redundant_exact_duplicates
    ]
    records = [record for record, _ in eligible]
    group_ids = [group_id for _, group_id in eligible]
    assign_grouped_splits(records, group_ids, fractions, seed)

    counts = {
        split: {
            label: sum(record["label"] == label and record["split"] == split for record in records)
            for label in labels
        }
        for split in SPLIT_NAMES
    }
    split_coverage = {
        label: sorted({
            record["split"] for record in records if record["label"] == label
        })
        for label in labels
    }
    group_counts = {
        label: len({group_ids[index] for index, record in enumerate(records) if record["label"] == label})
        for label in labels
    }
    return {
        "version": 1,
        "seed": seed,
        "dataset_dir": str(data_dir.resolve()),
        "split_fractions": fractions,
        "near_duplicate_hamming_threshold": near_distance,
        "class_names": labels,
        "image_count_valid": len(numeric_hash_records),
        "image_count_eligible": len(records),
        "rejected_corrupt_images": rejected,
        "removed_exact_duplicate_images": exact_duplicate_records,
        "excluded_cross_label_duplicate_images": [
            {"path": record["path"], "label": record["label"]} for record in excluded
        ],
        "counts_by_split_and_class": counts,
        "split_coverage_by_class": split_coverage,
        "independent_duplicate_groups_by_class": group_counts,
        "records": [
            {key: value for key, value in record.items() if key != "sha256" and key != "difference_hash"}
            | {"duplicate_group": group_id}
            for record, group_id in zip(records, group_ids)
        ],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data
    if not data_dir.exists():
        data_dir = Path(config["data_dir"])
    manifest = build_manifest(data_dir, config["split_fractions"], args.seed, args.near_duplicate_distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"valid={manifest['image_count_valid']} eligible={manifest['image_count_eligible']} "
        f"rejected={len(manifest['rejected_corrupt_images'])} "
        f"cross_label_duplicate_excluded={len(manifest['excluded_cross_label_duplicate_images'])}"
    )
    for split, counts in manifest["counts_by_split_and_class"].items():
        print(f"{split}={sum(counts.values())}")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
