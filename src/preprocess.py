import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import (
    ALLOWED_EXTENSIONS,
    CLEAN_DATASET_DIR,
    MIN_BLUR_SCORE,
    MIN_FACE_AREA_RATIO,
    MIN_IMAGE_SIDE,
    QUALITY_REPORT_CSV,
    RAW_DATASET_DIR,
)
from .face_engine import detect_faces, read_image


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_image(image_path: Path):
    image = read_image(image_path)
    h, w = image.shape[:2]

    if min(h, w) < MIN_IMAGE_SIDE:
        return False, "image_too_small"

    faces = detect_faces(image)
    if len(faces) != 1:
        return False, f"face_count_{len(faces)}"

    x1, y1, x2, y2 = map(int, faces[0].bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_area_ratio = max(0, x2-x1) * max(0, y2-y1) / float(w*h)

    if face_area_ratio < MIN_FACE_AREA_RATIO:
        return False, "face_too_small"

    score = blur_score(image)
    if score < MIN_BLUR_SCORE:
        return False, f"blurry_{score:.2f}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(RAW_DATASET_DIR))
    parser.add_argument("--output", default=str(CLEAN_DATASET_DIR))
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for person_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        out_person = output / person_dir.name
        out_person.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            try:
                accepted, reason = evaluate_image(image_path)
            except Exception as exc:
                accepted, reason = False, f"error_{type(exc).__name__}"

            if accepted:
                shutil.copy2(image_path, out_person / image_path.name)

            rows.append({
                "celebrity": person_dir.name,
                "image": image_path.name,
                "status": "accepted" if accepted else "rejected",
                "reason": reason,
            })

    QUALITY_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["celebrity", "image", "status", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(r["status"] == "accepted" for r in rows)
    rejected = len(rows) - accepted
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Report: {QUALITY_REPORT_CSV}")
    print(f"Clean dataset: {output}")


if __name__ == "__main__":
    main()
