import argparse
import csv
import re
import shutil
from pathlib import Path

from .config import ALLOWED_EXTENSIONS, KAGGLE_MANIFEST_CSV, RAW_DATASET_DIR


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "Unknown_Identity"


def image_files(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def find_identity_folders(source: Path):
    found = []
    for directory in sorted(p for p in source.rglob("*") if p.is_dir()):
        images = image_files(directory)
        if images:
            found.append((directory, images))
    return found


def main(source: Path, max_images: int):
    if not source.exists():
        raise FileNotFoundError(source)

    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    found = find_identity_folders(source)

    if not found:
        raise RuntimeError(
            "No image-containing folders were found. "
            "The downloaded dataset may use a different structure."
        )

    rows = []
    used_names = {}

    for identity_dir, images in found:
        identity = safe_name(identity_dir.name)
        used_names.setdefault(identity, 0)
        used_names[identity] += 1
        output_identity = identity if used_names[identity] == 1 else f"{identity}_{used_names[identity]}"

        out_dir = RAW_DATASET_DIR / output_identity
        out_dir.mkdir(parents=True, exist_ok=True)

        selected = images[:max_images]
        for i, source_image in enumerate(selected, start=1):
            destination = out_dir / f"{output_identity.lower()}_{i:02d}{source_image.suffix.lower()}"
            shutil.copy2(source_image, destination)
            rows.append({
                "source_folder": str(identity_dir),
                "source_image": source_image.name,
                "output_identity": output_identity,
                "output_image": destination.name,
            })

        print(f"[OK] {output_identity}: {len(selected)} selected / {len(images)} available")

    KAGGLE_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with KAGGLE_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_folder", "source_image", "output_identity", "output_image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared dataset: {RAW_DATASET_DIR}")
    print(f"Manifest: {KAGGLE_MANIFEST_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()
    main(Path(args.source), args.max_images)
