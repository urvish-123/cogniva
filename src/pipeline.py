"""
Unified recognition. Two flows, chosen by whether a patient_id is given:

A. With `patient_id` (patient flow): ONLY that patient's people, from
   dataset/Patient/<patient_id>/<Person>/ (see src/patient.py).
   - best match >= 80%               -> type "patient"
   - otherwise (no match, no face,
     multiple faces, no people)      -> type "unknown"
   Celebrities and objects are never searched. An unknown patient_id raises
   PatientNotFound (HTTP 404 in the API).

B. Without `patient_id` (general flow):
   1. Face recognition runs first.
      - celebrity best match >= 80%  -> type "celebrity"
      - no match                     -> type "unknown" (CLIP is skipped)
      - multiple faces               -> type "unknown"
   2. No face -> CLIP object recognition.
      - object match                 -> type "object"
      - otherwise                    -> type "unknown"

Every result has the keys type, patient, celebrity, object (null when that
step did not run) and `processing_time` in milliseconds.
"""

import argparse
import time
from pathlib import Path

from .patient import recognize_patient
from .object_recognize import recognize_object
from .recognize import recognize_image


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 1)


def recognize(image_path: Path, patient_id=None):
    if patient_id:
        return recognize_for_patient(image_path, patient_id)

    started = time.perf_counter()
    timings = {}

    step = time.perf_counter()
    celebrity = recognize_image(image_path)
    timings["face_ms"] = elapsed_ms(step)

    obj = None
    if celebrity["status"] == "match":
        kind = "celebrity"
    elif celebrity["status"] in ("unknown", "multiple_faces"):
        kind = "unknown"
    else:
        step = time.perf_counter()
        obj = recognize_object(image_path)
        timings["object_ms"] = elapsed_ms(step)
        kind = "object" if obj["status"] == "match" else "unknown"

    timings["total_ms"] = elapsed_ms(started)

    return {
        "type": kind,
        "patient": None,
        "celebrity": celebrity,
        "object": obj,
        "processing_time": timings,
    }


def recognize_for_patient(image_path: Path, patient_id):
    started = time.perf_counter()
    patient = recognize_patient(image_path, patient_id)
    face_ms = elapsed_ms(started)

    return {
        "type": "patient" if patient["status"] == "match" else "unknown",
        "patient": patient,
        "celebrity": None,
        "object": None,
        "processing_time": {"face_ms": face_ms, "total_ms": face_ms},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--patient-id", default=None)
    args = parser.parse_args()

    print(recognize(Path(args.image), args.patient_id))
