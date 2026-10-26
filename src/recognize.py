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


class Gallery:
    """Records with their centroids stacked and normalized once, so a search
    is a single matrix product instead of a Python loop over the records."""

    def __init__(self, records):
        self.records = records
        if records:
            matrix = np.stack([r["centroid"] for r in records]).astype(np.float32)
            self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        else:
            self.matrix = np.empty((0, 0), dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def rank(self, query, top=5):
        """(cosine score, record) of the `top` best records, best first."""
        query = np.asarray(query, dtype=np.float32)
        scores = self.matrix @ (query / np.linalg.norm(query))
        order = np.argsort(-scores)[:top]
        return [(float(scores[i]), self.records[i]) for i in order]


# Rebuilt only when the database returns a new list (i.e. it changed).
_celebrities = Gallery([])


def celebrity_gallery():
    global _celebrities
    records = get_all_celebrities()
    if _celebrities.records is not records:
        _celebrities = Gallery(records)
    return _celebrities


def face_query(image):
    """(embedding, None) when the image has exactly one face, otherwise
    (None, result) with status "no_face" or "multiple_faces"."""
    # Detect first; the costly embedding runs only for a single-face image.
    faces = detect_faces(image, embed=False)

    if len(faces) == 0:
        return None, {
            "status": "no_face",
            "message": "No face detected.",
        }

    if len(faces) > 1:
        return None, {
            "status": "multiple_faces",
            "message": f"{len(faces)} faces detected. Please use one face.",
        }

    return normalized_embedding(faces[0], image), None


def recognize_image(image_path: Path, threshold=SIMILARITY_THRESHOLD):
    query, failure = face_query(read_image(image_path))
    if failure:
        return failure
    return recognize_celebrity(query, threshold)


def recognize_celebrity(query, threshold=SIMILARITY_THRESHOLD):
    gallery = celebrity_gallery()

    if not gallery:
        return {
            "status": "database_empty",
            "message": "No celebrity embeddings are stored in the database.",
        }

    scored = gallery.rank(query)
    top = [{"name": c["name"], "score": round(s, 4)} for s, c in scored]
    score, best = scored[0]

    if score >= threshold:
        return {
            "status": "match",
            "name": best["name"],
            "category": best.get("category", ""),
            "description": best.get("description", ""),
            "details": best.get("details", {}),
            "score": round(score, 4),
            "threshold": threshold,
            "sample_count": best["sample_count"],
            "top_matches": top,
        }

    return {
        "status": "unknown",
        "name": "Unknown",
        "score": round(score, 4),
        "threshold": threshold,
        "top_matches": top,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    print(recognize_image(Path(args.image), args.threshold))
