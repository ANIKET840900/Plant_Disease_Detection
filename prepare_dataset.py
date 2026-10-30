"""Download and prepare the official PlantVillage color images for training."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/PlantVillage"))
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for a smoke-test subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename="data.zip",
        repo_type="dataset",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as source:
        members = (
            member
            for member in source.infolist()
            if not member.is_dir() and member.filename.replace("\\", "/").startswith("raw/color/")
        )
        for member in members:
            parts = Path(member.filename.replace("\\", "/")).parts
            if len(parts) < 4:
                continue
            class_name = parts[2]
            filename = Path(parts[-1]).name
            if not class_name or not filename:
                continue
            target = (args.output / class_name / filename).resolve()
            if args.output.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
            extracted += 1
            if args.max_images and extracted >= args.max_images:
                break
    if extracted == 0:
        raise SystemExit("No color images were found in the downloaded PlantVillage archive.")
    print(f"prepared_images={extracted} output={args.output}")


if __name__ == "__main__":
    main()