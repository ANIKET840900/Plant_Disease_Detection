"""Fetch PlantDoc images without checking invalid Windows filenames out to disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SOURCE_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
INVALID_WINDOWS_CHARACTERS = re.compile(r'[<>:"/\\|?*]')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("data/raw/PlantDoc"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/PlantDoc_dataset"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_git(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        stderr=subprocess.PIPE,
    )


def ensure_repository(repo: Path) -> None:
    if (repo / ".git").exists():
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--no-checkout", SOURCE_URL, str(repo)],
        check=True,
    )


def list_dataset_blobs(repo: Path) -> list[tuple[str, str]]:
    tree = run_git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    entries: list[tuple[str, str]] = []
    for item in tree.split(b"\0"):
        if not item:
            continue
        header, path_bytes = item.split(b"\t", 1)
        _, kind, object_id = header.decode("ascii").split()
        name = path_bytes.decode("utf-8", "surrogateescape")
        is_dataset_file = name.startswith(("train/", "test/"))
        is_source_document = name in {"README.md", "LICENSE.txt"}
        if kind == "blob" and (is_dataset_file or is_source_document):
            entries.append((object_id, name))
    return entries


def _destination_for_path(output: Path, source_path: str, object_id: str) -> Path:
    safe_parts = [
        INVALID_WINDOWS_CHARACTERS.sub("_", part).rstrip(" .")
        for part in source_path.split("/")
    ]
    destination = output.joinpath(*safe_parts)
    if destination.exists():
        destination = destination.with_name(
            f"{destination.stem}_{object_id[:8]}{destination.suffix}"
        )
    return destination


def extract_repository(repo: Path, output: Path, force: bool = False) -> dict[str, Any]:
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source") == SOURCE_URL:
            return manifest
        raise FileExistsError(
            f"Output exists with a different dataset manifest: {manifest_path}. "
            "Choose a new --output directory or pass --force."
        )

    entries = list_dataset_blobs(repo)
    image_entries = [
        (object_id, name)
        for object_id, name in entries
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_entries:
        raise ValueError(f"No supported PlantDoc images were found in {repo}.")

    input_data = "".join(f"{object_id}\n" for object_id, _ in entries).encode("ascii")
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_bytes, error_bytes = process.communicate(input=input_data)
    if process.returncode:
        raise RuntimeError(error_bytes.decode("utf-8", "replace"))

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    offset = 0
    for object_id, source_path in entries:
        header_end = output_bytes.find(b"\n", offset)
        if header_end < 0:
            raise RuntimeError(f"Missing Git blob header for {source_path}.")
        header = output_bytes[offset:header_end].decode("ascii").split()
        if (
            len(header) != 3
            or header[0] != object_id
            or header[1] != "blob"
        ):
            raise RuntimeError(f"Unexpected Git blob response for {source_path}.")
        size = int(header[2])
        content_start = header_end + 1
        content_end = content_start + size
        contents = output_bytes[content_start:content_end]
        offset = content_end + 1

        if source_path in {"README.md", "LICENSE.txt"}:
            (output / source_path).write_bytes(contents)
            continue
        if Path(source_path).suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        destination = _destination_for_path(output, source_path, object_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        try:
            with Image.open(destination) as image:
                image.verify()
        except (
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            ValueError,
        ) as error:
            rejected.append(
                {"source_path": source_path, "reason": type(error).__name__}
            )
            continue

        path_parts = source_path.split("/", 2)
        records.append(
            {
                "source_path": source_path,
                "path": destination.relative_to(output).as_posix(),
                "split": path_parts[0],
                "class_name": path_parts[1],
                "sha256": hashlib.sha256(contents).hexdigest(),
                "bytes": size,
            }
        )

    manifest: dict[str, Any] = {
        "source": SOURCE_URL,
        "commit": run_git(repo, "rev-parse", "HEAD").decode("ascii").strip(),
        "license": "CC BY 4.0; see LICENSE.txt and README.md.",
        "image_count_downloaded": len(image_entries),
        "valid_images": len(records),
        "rejected_images": rejected,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    ensure_repository(args.repo)
    manifest = extract_repository(args.repo, args.output, args.force)
    records = manifest["records"]
    print(
        f"downloaded={manifest['image_count_downloaded']} "
        f"valid={manifest['valid_images']} rejected={len(manifest['rejected_images'])}"
    )
    for split in ("train", "test"):
        split_records = [record for record in records if record["split"] == split]
        class_count = len({record["class_name"] for record in split_records})
        print(f"{split}={len(split_records)} classes={class_count}")
    print(f"manifest={args.output / 'source_manifest.json'}")


if __name__ == "__main__":
    main()
