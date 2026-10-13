import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import (
    ALLOWED_EXTENSIONS,
    CLEAN_DATASET_DIR,
    MIN_IMAGES_PER_IDENTITY,
    RAW_DATASET_DIR,
    TEST_CASES_CSV,
    TEST_IMAGES_DIR,
    UNKNOWN_PEOPLE_DIR,
    UNKNOWN_TEST_IMAGES,
)
from .database import clear_celebrities, init_database, upsert_celebrity
from .face_engine import detect_faces, normalized_embedding, read_image
from .metadata import load_metadata, name_key


def image_files(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def copy_test_image(source: Path, name: str):
    destination = TEST_IMAGES_DIR / name
    shutil.copy2(source, destination)
    return destination


def side_by_side(paths):
    images = [cv2.resize(read_image(p), (160, 160)) for p in paths]
    return np.hstack(images)


def add_dataset_tests(test_cases, samples, metadata):
    # Test 1: every image in the celebrity dataset.
    if not RAW_DATASET_DIR.exists():
        return
    for person_dir in sorted(p for p in RAW_DATASET_DIR.iterdir() if p.is_dir()):
        meta = metadata.get(name_key(person_dir.name))
        if not meta:
            continue
        images = image_files(person_dir)
        for image_path in images:
            test_cases.append((image_path, meta["name"], "Celebrity dataset image"))
        if images:
            samples.append((meta["name"], images[0]))


def write_test_cases(test_cases, samples):
    # Test 2: different lighting -> darkened copy of a dataset image.
    if samples:
        name, source = samples[0]
        dark = cv2.convertScaleAbs(read_image(source), alpha=0.5, beta=0)
        path = TEST_IMAGES_DIR / "lighting_dark_test.jpg"
        cv2.imwrite(str(path), dark)
        test_cases.append((path, name, "Different lighting"))

    # Test 5: multiple faces -> two dataset faces of different people.
    if len(samples) >= 2:
        first = samples[0][1]
        second = next(p for n, p in samples if n != samples[0][0])
        path = TEST_IMAGES_DIR / "multiple_faces_test.jpg"
        cv2.imwrite(str(path), side_by_side([first, second]))
        test_cases.append((path, "multiple_faces", "Multiple faces"))

    root = TEST_IMAGES_DIR.parent
    with TEST_CASES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "expected_name", "case"])
        for path, expected, case in test_cases:
            writer.writerow([path.relative_to(root).as_posix(), expected, case])

    print(f"Test cases: {len(test_cases)} written to {TEST_CASES_CSV}")


def clear_generated_test_images():
    for pattern in ("*_test_*.jpg", "unknown_*.jpg"):
        for path in TEST_IMAGES_DIR.glob(pattern):
            path.unlink()


def add_unknown_tests(test_cases):
    # Test 4: unknown person -> people who are never stored in the database.
    if not UNKNOWN_PEOPLE_DIR.exists():
        return
    for person_dir in sorted(p for p in UNKNOWN_PEOPLE_DIR.iterdir() if p.is_dir()):
        selected = 0
        for image_path in image_files(person_dir):
            if selected == UNKNOWN_TEST_IMAGES:
                break
            # These photos are not cleaned; keep only single-face ones.
            if len(detect_faces(read_image(image_path))) != 1:
                continue
            selected += 1
            path = copy_test_image(
                image_path, f"unknown_{person_dir.name.lower()}_{selected:02d}.jpg")
            test_cases.append((path, "Unknown", "Unknown person"))


def main(dataset: Path):
    init_database()
    clear_celebrities()
    metadata = load_metadata()
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    clear_generated_test_images()

    test_cases = []
    samples = []
    total_people = 0
    too_few = []

    for person_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        images = image_files(person_dir)
        meta = metadata.get(name_key(person_dir.name))

        # Only people listed in the celebrity metadata go into the database.
        if not meta:
            print(f"[SKIP] {person_dir.name}: not in celebrity metadata")
            continue

        # Exact spelling from the metadata (e.g. "P. V. Sindhu").
        name = meta["name"]

        if len(images) < MIN_IMAGES_PER_IDENTITY:
            too_few.append(name)
            continue

        embeddings = []
        skipped = 0

        for image_path in images:
            try:
                image = read_image(image_path)
                faces = detect_faces(image)

                if len(faces) != 1:
                    skipped += 1
                    continue

                embeddings.append(normalized_embedding(faces[0]))
            except Exception:
                skipped += 1

        if not embeddings:
            print(f"[SKIP] {person_dir.name}: no valid embeddings")
            continue

        matrix = np.vstack(embeddings)
        centroid = np.mean(matrix, axis=0)
        centroid /= np.linalg.norm(centroid)

        upsert_celebrity(
            name=name,
            category=meta.get("category", ""),
            language=meta.get("language", ""),
            description=meta.get("description", ""),
            details=meta.get("details", {}),
            centroid=centroid,
            sample_count=len(embeddings),
        )

        total_people += 1
        print(
            f"[OK] {name}: {len(embeddings)} embeddings, "
            f"{skipped} skipped, dimension={centroid.shape[0]}"
        )

    print(f"Database built for {total_people} identities.")
    if too_few:
        print(f"{len(too_few)} celebrities skipped (fewer than "
              f"{MIN_IMAGES_PER_IDENTITY} images). Run: python -m src.metadata")

    add_dataset_tests(test_cases, samples, metadata)
    add_unknown_tests(test_cases)
    write_test_cases(test_cases, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(CLEAN_DATASET_DIR))
    args = parser.parse_args()
    main(Path(args.dataset))
