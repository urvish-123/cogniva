import argparse
from pathlib import Path

import numpy as np

from .config import OBJECT_SIMILARITY_THRESHOLD
from .database import get_all_objects
from .object_engine import image_embedding


def recognize_object(image_path: Path, threshold=OBJECT_SIMILARITY_THRESHOLD):
    objects = get_all_objects()

    if not objects:
        return {
            "status": "database_empty",
            "message": "No object embeddings are stored in the database.",
        }

    query = image_embedding(image_path)
    # Both sides are L2-normalized, so the dot product is cosine similarity.
    scores = np.vstack([obj["centroid"] for obj in objects]) @ query
    order = np.argsort(scores)[::-1]
    best = objects[order[0]]
    best_score = float(scores[order[0]])

    top_matches = [
        {"name": objects[i]["name"], "score": round(float(scores[i]), 4)}
        for i in order[:5]
    ]

    if best_score > threshold:
        return {
            "status": "match",
            "name": best["name"],
            "category": best["category"],
            "score": round(best_score, 4),
            "threshold": threshold,
            "sample_count": best["sample_count"],
            "top_matches": top_matches,
        }

    return {
        "status": "unknown",
        "name": "Unknown",
        "score": round(best_score, 4),
        "threshold": threshold,
        "top_matches": top_matches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--threshold", type=float, default=OBJECT_SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    print(recognize_object(Path(args.image), args.threshold))
