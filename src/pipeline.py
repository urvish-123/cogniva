"""
Unified recognition: celebrity (InsightFace / ArcFace) or object (CLIP).

1. Face recognition runs first (existing celebrity logic, unchanged).
   - celebrity match   -> type "celebrity"
   - multiple faces    -> type "unknown" (one face per image, as before)
2. No face, or a face that matches no celebrity -> CLIP object recognition.
   - object match      -> type "object"
   - otherwise         -> type "unknown"

Every result carries `processing_time` in milliseconds.
"""

import argparse
import time
from pathlib import Path

from .object_recognize import recognize_object
from .recognize import recognize_image


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 1)


def recognize(image_path: Path):
    started = time.perf_counter()
    timings = {}

    step = time.perf_counter()
    celebrity = recognize_image(image_path)
    timings["face_ms"] = elapsed_ms(step)

    obj = None
    if celebrity["status"] == "match":
        kind = "celebrity"
    elif celebrity["status"] == "multiple_faces":
        kind = "unknown"
    else:
        step = time.perf_counter()
        obj = recognize_object(image_path)
        timings["object_ms"] = elapsed_ms(step)
        kind = "object" if obj["status"] == "match" else "unknown"

    timings["total_ms"] = elapsed_ms(started)

    return {
        "type": kind,
        "celebrity": celebrity,
        "object": obj,
        "processing_time": timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    print(recognize(Path(args.image)))
