"""
Build one CLIP centroid per object class and store it in the `objects` table.

python -m src.generate_object_centroids
"""

import argparse
from pathlib import Path

import numpy as np

from .config import ALLOWED_EXTENSIONS, OBJECT_METADATA_XLSX, OBJECT_TRAIN_DIR
from .database import clear_objects, init_database, upsert_object
from .metadata import name_key
from .object_engine import image_embedding


def load_object_categories():
    """{name_key: (display name, category)} from metadata/objects.xlsx."""
    if not OBJECT_METADATA_XLSX.exists():
        return {}

    from openpyxl import load_workbook

    sheet = load_workbook(OBJECT_METADATA_XLSX, read_only=True).worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(h or "").strip().lower() for h in next(rows)]
    name_col, category_col = headers.index("object"), headers.index("category")

    categories = {}
    for row in rows:
        name = str(row[name_col] or "").strip()
        if name:
            category = str(row[category_col] or "").strip()
            categories[name_key(name)] = (name, category)
    return categories


def main(dataset: Path):
    init_database()
    clear_objects()
    categories = load_object_categories()
    total = 0

    for object_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        images = sorted(
            p for p in object_dir.iterdir()
            if p.suffix.lower() in ALLOWED_EXTENSIONS
        )
        # "Air_Conditioner" folder -> "Air Conditioner" from the Excel sheet.
        fallback = object_dir.name.replace("_", " ")
        name, category = categories.get(
            name_key(object_dir.name), (fallback, "Unknown"))

        embeddings = []
        for image_path in images:
            try:
                embeddings.append(image_embedding(image_path))
            except Exception as exc:
                print(f"  [FAIL] {image_path.name}: {exc}")

        if not embeddings:
            print(f"[SKIP] {name}: no valid embeddings")
            continue

        centroid = np.mean(np.vstack(embeddings), axis=0)
        centroid /= np.linalg.norm(centroid)

        upsert_object(
            name=name,
            category=category,
            centroid=centroid,
            sample_count=len(embeddings),
        )
        total += 1
        print(f"[OK] {name} ({category}): {len(embeddings)} embeddings, "
              f"dimension={centroid.shape[0]}")

    print(f"Object database built for {total} classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(OBJECT_TRAIN_DIR))
    args = parser.parse_args()
    main(Path(args.dataset))
