import argparse
from pathlib import Path

import numpy as np

from .config import SIMILARITY_THRESHOLD
from .database import get_all_celebrities
from .face_engine import detect_faces, normalized_embedding, read_image


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def recognize_image(image_path: Path, threshold=SIMILARITY_THRESHOLD):
    image = read_image(image_path)
    faces = detect_faces(image)

    if len(faces) == 0:
        return {
            "status": "no_face",
            "message": "No face detected.",
        }

    if len(faces) > 1:
        return {
            "status": "multiple_faces",
            "message": f"{len(faces)} faces detected. Please use one face.",
        }

    query = normalized_embedding(faces[0])
    celebrities = get_all_celebrities()

    if not celebrities:
        return {
            "status": "database_empty",
            "message": "No celebrity embeddings are stored in the database.",
        }

    scored = []
    for celebrity in celebrities:
        score = cosine_similarity(query, celebrity["centroid"])
        scored.append({
            "name": celebrity["name"],
            "score": score,
            "sample_count": celebrity["sample_count"],
            "category": celebrity.get("category", ""),
            "description": celebrity.get("description", ""),
            "details": celebrity.get("details", {}),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    if best["score"] > threshold:
        return {
            "status": "match",
            "name": best["name"],
            "category": best["category"],
            "description": best["description"],
            "details": best["details"],
            "score": round(best["score"], 4),
            "threshold": threshold,
            "sample_count": best["sample_count"],
            "top_matches": [
                {"name": x["name"], "score": round(x["score"], 4)}
                for x in scored[:5]
            ],
        }

    return {
        "status": "unknown",
        "name": "Unknown",
        "score": round(best["score"], 4),
        "threshold": threshold,
        "top_matches": [
            {"name": x["name"], "score": round(x["score"], 4)}
            for x in scored[:5]
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    print(recognize_image(Path(args.image), args.threshold))
