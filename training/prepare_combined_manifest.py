"""Merge PlantVillage splits with PlantDoc train images, excluding duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from training.prepare_splits import hamming_distance, image_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantvillage", type=Path, default=Path("data/processed/PlantVillage"))
    parser.add_argument(
        "--plantvillage-manifest",
        type=Path,
        default=Path("reports/generated/split_manifest.json"),
    )
    parser.add_argument("--plantdoc", type=Path, default=Path("data/raw/PlantDoc_dataset"))
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("training/plantdoc_label_mapping.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/combined_split_manifest.json"),
    )
    parser.add_argument(
        "--hash-cache",
        type=Path,
        default=Path("reports/generated/combined_hash_cache.json"),
    )
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    return parser.parse_args()


def _record_path(data_dir: Path, record: dict[str, Any]) -> Path:
    root = Path(record.get("dataset_root", "."))
    return data_dir / root / Path(record["path"])


def _validated_hash(path: Path) -> tuple[str, int]:
    try:
        return image_hashes(path)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        ValueError,
    ) as error:
        raise ValueError(f"Unable to validate image {path}: {error}") from error


def _cached_hash(path: Path, cache: dict[str, Any]) -> tuple[str, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    key = str(resolved)
    cached = cache.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
        and isinstance(cached.get("difference_hash"), int)
    ):
        return cached["sha256"], cached["difference_hash"]
    digest, perceptual_hash = _validated_hash(resolved)
    cache[key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "difference_hash": perceptual_hash,
    }
    return digest, perceptual_hash


def _has_valid_cached_hash(path: Path, cache: dict[str, Any]) -> bool:
    resolved = path.resolve()
    stat = resolved.stat()
    cached = cache.get(str(resolved))
    return (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
        and isinstance(cached.get("difference_hash"), int)
    )


def _save_hash_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(cache), encoding="utf-8")
    temporary_path.replace(path)


def build_combined_manifest(
    plantvillage_dir: Path,
    plantvillage_manifest: dict[str, Any],
    plantdoc_dir: Path,
    plantdoc_manifest: dict[str, Any],
    mapping: dict[str, str],
    near_duplicate_distance: int = 3,
    hash_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Append mapped PlantDoc train images without leaking duplicates across splits."""
    if not 0 <= near_duplicate_distance <= 3:
        raise ValueError("Near-duplicate Hamming distance must be between zero and three.")
    records = [dict(record) for record in plantvillage_manifest["records"]]
    classes = list(plantvillage_manifest["class_names"])
    class_set = set(classes)
    if set(mapping.values()) - class_set:
        raise ValueError("PlantDoc mapping references classes absent from PlantVillage.")

    exact_hashes: dict[str, set[str]] = defaultdict(set)
    hash_buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    hash_cache: dict[str, Any] = {}
    if hash_cache_path is not None and hash_cache_path.is_file():
        loaded_cache = json.loads(hash_cache_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_cache, dict):
            raise ValueError(f"Image hash cache must be a JSON object: {hash_cache_path}")
        hash_cache = loaded_cache
    dirty_cache = False
    next_group = max(
        (int(record.get("duplicate_group", -1)) for record in records),
        default=-1,
    ) + 1
    for index, record in enumerate(records, start=1):
        image_path = _record_path(plantvillage_dir, record)
        was_cached = (
            hash_cache_path is not None
            and _has_valid_cached_hash(image_path, hash_cache)
        )
        digest, perceptual_hash = (
            _cached_hash(image_path, hash_cache)
            if hash_cache_path is not None
            else _validated_hash(image_path)
        )
        dirty_cache = dirty_cache or not was_cached
        exact_hashes[digest].add(record["label"])
        for chunk in range(4):
            bucket_value = (perceptual_hash >> (chunk * 16)) & 0xFFFF
            hash_buckets[(chunk, bucket_value)].append(
                (perceptual_hash, record["label"])
            )
        if index % 5000 == 0:
            print(
                f"checked_plantvillage={index}/{len(records)}",
                flush=True,
            )
            if hash_cache_path is not None and dirty_cache:
                _save_hash_cache(hash_cache_path, hash_cache)
                dirty_cache = False

    source_counts = {
        "plantvillage": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "validation", "calibration", "test")
        },
        "plantdoc_train_added": 0,
    }
    excluded = {
        "plantdoc_unsupported_label": [],
        "plantdoc_duplicate_or_conflicting_label": [],
        "plantdoc_near_duplicate_of_plantvillage": [],
        "plantdoc_duplicate_within_training": [],
        "plantdoc_corrupt_or_missing": [],
    }
    accepted_exact: dict[str, str] = {}
    accepted_buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    plantdoc_records = plantdoc_manifest.get("records", [])

    field_train_total = sum(
        record.get("split") == "train" for record in plantdoc_records
    )
    field_train_scanned = 0
    for field_record in plantdoc_records:
        if field_record.get("split") != "train":
            continue
        field_train_scanned += 1
        original_label = field_record.get("class_name", "")
        label = mapping.get(original_label)
        if label not in class_set:
            excluded["plantdoc_unsupported_label"].append(field_record["source_path"])
            continue
        image_path = (plantdoc_dir / field_record["path"]).resolve()
        if plantdoc_dir.resolve() not in image_path.parents:
            raise ValueError(
                f"PlantDoc image path escapes its data directory: {field_record['path']}"
            )
        try:
            was_cached = (
                hash_cache_path is not None
                and _has_valid_cached_hash(image_path, hash_cache)
            )
            digest, perceptual_hash = (
                _cached_hash(image_path, hash_cache)
                if hash_cache_path is not None
                else _validated_hash(image_path)
            )
        except ValueError:
            excluded["plantdoc_corrupt_or_missing"].append(
                field_record["source_path"]
            )
            continue
        dirty_cache = dirty_cache or not was_cached

        if digest in exact_hashes:
            excluded["plantdoc_duplicate_or_conflicting_label"].append(
                {
                    "path": field_record["source_path"],
                    "plantdoc_label": label,
                    "matching_plantvillage_labels": sorted(exact_hashes[digest]),
                }
            )
            continue

        matching_labels: set[str] = set()
        for chunk in range(4):
            bucket_value = (perceptual_hash >> (chunk * 16)) & 0xFFFF
            for candidate_hash, candidate_label in hash_buckets[
                (chunk, bucket_value)
            ]:
                if hamming_distance(perceptual_hash, candidate_hash) <= near_duplicate_distance:
                    matching_labels.add(candidate_label)
        if matching_labels:
            excluded["plantdoc_near_duplicate_of_plantvillage"].append(
                {
                    "path": field_record["source_path"],
                    "plantdoc_label": label,
                    "matching_plantvillage_labels": sorted(matching_labels),
                }
            )
            continue

        if digest in accepted_exact:
            excluded["plantdoc_duplicate_within_training"].append(
                field_record["source_path"]
            )
            continue
        duplicate_found = False
        for chunk in range(4):
            bucket_value = (perceptual_hash >> (chunk * 16)) & 0xFFFF
            for candidate_hash, _ in accepted_buckets[(chunk, bucket_value)]:
                if (
                    hamming_distance(perceptual_hash, candidate_hash)
                    <= near_duplicate_distance
                ):
                    duplicate_found = True
                    break
            if duplicate_found:
                break
        if duplicate_found:
            excluded["plantdoc_duplicate_within_training"].append(
                field_record["source_path"]
            )
            continue

        records.append(
            {
                "path": field_record["path"],
                "dataset_root": str(
                    Path("..") / ".." / "raw" / "PlantDoc_dataset"
                ),
                "label": label,
                "split": "train",
                "duplicate_group": next_group,
                "source": "PlantDoc",
            }
        )
        next_group += 1
        source_counts["plantdoc_train_added"] += 1
        accepted_exact[digest] = label
        for chunk in range(4):
            bucket_value = (perceptual_hash >> (chunk * 16)) & 0xFFFF
            accepted_buckets[(chunk, bucket_value)].append(
                (perceptual_hash, label)
            )
        if field_train_scanned % 500 == 0:
            print(
                f"hashed_plantdoc_train={field_train_scanned}/{field_train_total}",
                flush=True,
            )
            if hash_cache_path is not None and dirty_cache:
                _save_hash_cache(hash_cache_path, hash_cache)
                dirty_cache = False

    if hash_cache_path is not None and dirty_cache:
        _save_hash_cache(hash_cache_path, hash_cache)

    split_counts = {
        split: {
            label: sum(
                record["split"] == split and record["label"] == label
                for record in records
            )
            for label in classes
        }
        for split in ("train", "validation", "calibration", "test")
    }
    return {
        "version": 2,
        "seed": plantvillage_manifest["seed"],
        "dataset_dir": str(plantvillage_dir.resolve()),
        "plantvillage_manifest": plantvillage_manifest.get("dataset_dir"),
        "plantdoc_source": plantdoc_manifest.get("source"),
        "plantdoc_commit": plantdoc_manifest.get("commit"),
        "plantdoc_license": plantdoc_manifest.get("license"),
        "class_names": classes,
        "split_fractions": plantvillage_manifest["split_fractions"],
        "source_counts": source_counts,
        "excluded_plantdoc_records": excluded,
        "counts_by_split_and_class": split_counts,
        "records": records,
        "holdout_note": (
            "PlantDoc official test images are not added to this training manifest. "
            "Validation, calibration and test remain PlantVillage-only."
        ),
    }


def main() -> None:
    args = parse_args()
    pv_manifest = json.loads(
        args.plantvillage_manifest.read_text(encoding="utf-8")
    )
    field_manifest = json.loads(
        (args.plantdoc / "source_manifest.json").read_text(encoding="utf-8")
    )
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    manifest = build_combined_manifest(
        args.plantvillage,
        pv_manifest,
        args.plantdoc,
        field_manifest,
        mapping,
        args.near_duplicate_distance,
        args.hash_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"plantvillage_train={manifest['source_counts']['plantvillage']['train']} "
        f"plantdoc_train_added={manifest['source_counts']['plantdoc_train_added']} "
        f"combined_train={sum(manifest['counts_by_split_and_class']['train'].values())}"
    )
    print(
        f"excluded_exact_or_conflicting="
        f"{len(manifest['excluded_plantdoc_records']['plantdoc_duplicate_or_conflicting_label'])} "
        f"excluded_near_duplicate="
        f"{len(manifest['excluded_plantdoc_records']['plantdoc_near_duplicate_of_plantvillage'])} "
        f"excluded_within_plantdoc="
        f"{len(manifest['excluded_plantdoc_records']['plantdoc_duplicate_within_training'])}"
    )
    print(
        "holdout_counts="
        + json.dumps(
            {
                split: sum(manifest["counts_by_split_and_class"][split].values())
                for split in ("validation", "calibration", "test")
            }
        )
    )
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
