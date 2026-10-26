"""
Patient recognition: match a face against ONE patient's people only.

Photos: dataset/Patient/<patient_id>/<Person>/*.jpg
        e.g. dataset/Patient/P001/Son_of_amitab/1.jpg
             dataset/Patient/P001/Wife_of_amitab/1.jpg

The patient folder name is the patient_id (case-insensitive: "p001" finds
P001); the person folder name is the name shown, with underscores as spaces
("Son of amitab"). Celebrities and objects are never searched here, so a
request costs one face embedding plus one small matrix product.

Indexing is automatic and incremental:
- each person's centroid is stored in `patient_people` with the folder's
  modification time; when photos are added, removed or renamed in a person
  folder, only that person is embedded again, and removed folders are dropped;
- a patient whose folders did not change is served from memory, without
  SQLite or reading any photo (a request only lists the person folders).
Replacing a photo in place under the same file name does not change the
folder time; add it under a new name, or run
`python -m src.generate_patient_centroids --rebuild`.
"""

import os
import re
import threading
from pathlib import Path

from .config import (
    PATIENT_DATASET_DIR,
    PATIENT_SIMILARITY_THRESHOLD,
    MIN_IMAGES_PER_PATIENT_PERSON,
)
from .database import (
    delete_patient_people,
    get_patient_people,
    upsert_patient_person,
)
from .face_engine import read_image
from .generate_centroids import folder_centroid, image_files
from .recognize import Gallery, face_query


class PatientNotFound(LookupError):
    """No folder dataset/Patient/<patient_id>."""


# Letters, digits, "_" and "-" only: an ID can never lead outside the folder.
_PATIENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def patient_key(patient_id):
    """Canonical patient ID ("p001 " -> "P001"), or None if malformed."""
    patient_id = (patient_id or "").strip()
    if not _PATIENT_ID.fullmatch(patient_id):
        return None
    return patient_id.upper()


def patient_folder(patient_id):
    """The patient's folder, or None when there is none."""
    key = patient_key(patient_id)
    if key is None:
        return None
    path = PATIENT_DATASET_DIR / key
    if path.is_dir():
        return path
    # Case-sensitive file systems: a folder named e.g. "p001".
    try:
        with os.scandir(PATIENT_DATASET_DIR) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.upper() == key:
                    return Path(entry.path)
    except FileNotFoundError:
        pass
    return None


def person_folders(folder: Path):
    """{person folder name: modification time} for one patient."""
    # os.stat, not DirEntry.stat(): on Windows the latter returns the copy
    # cached in the parent folder, which lags behind changes in the folder.
    with os.scandir(folder) as entries:
        return {e.name: os.stat(e.path).st_mtime for e in entries
                if e.is_dir() and not e.name.startswith(".")}


def sync_patient(key, folder: Path, folders=None, verbose=False):
    """Bring the patient's index in line with its folder, embedding only
    new or changed people. Returns the indexed people."""
    if folders is None:
        folders = person_folders(folder)
    indexed = {p["folder"]: p for p in get_patient_people(key)}

    removed = [name for name in indexed if name not in folders]
    if removed:
        delete_patient_people(key, removed)

    for name, mtime in sorted(folders.items()):
        row = indexed.get(name)
        if row is not None and row["folder_mtime"] == mtime:
            continue

        images = image_files(folder / name)
        centroid, count, skipped = (None, 0, len(images))
        if len(images) >= MIN_IMAGES_PER_PATIENT_PERSON:
            centroid, count, skipped = folder_centroid(images)

        if centroid is None:
            if row is not None:
                delete_patient_people(key, [name])
            if verbose:
                print(f"[SKIP] {key} / {name}: {len(images)} photos, "
                      "no usable single-face photo")
            continue

        upsert_patient_person(key, name, centroid, count, mtime)
        if verbose:
            print(f"[OK] {key} / {name}: {count} embeddings, {skipped} skipped")

    return get_patient_people(key)


# patient key -> (person folder times it was built from, Gallery)
_galleries = {}
_lock = threading.Lock()


def patient_gallery(key, folder: Path):
    """The patient's people, ready to search; re-synced only when its
    person folders changed since the last request."""
    folders = person_folders(folder)
    cached = _galleries.get(key)
    if cached is not None and cached[0] == folders:
        return cached[1]

    with _lock:  # one sync per patient at a time
        cached = _galleries.get(key)
        if cached is not None and cached[0] == folders:
            return cached[1]
        gallery = Gallery(sync_patient(key, folder, folders))
        _galleries[key] = (folders, gallery)
        return gallery


def load_all_patients(verbose=False):
    """Index every patient folder and keep it in memory (API startup).
    Returns the number of people ready to match."""
    if not PATIENT_DATASET_DIR.is_dir():
        return 0
    total = 0
    for folder in sorted(p for p in PATIENT_DATASET_DIR.iterdir() if p.is_dir()):
        key = patient_key(folder.name)
        if key is None:
            continue
        if verbose:
            print(f"Patient {key}")
        with _lock:
            folders = person_folders(folder)
            gallery = Gallery(sync_patient(key, folder, folders, verbose))
            _galleries[key] = (folders, gallery)
        total += len(gallery)
    return total


def recognize_patient(image_path, patient_id,
                      threshold=PATIENT_SIMILARITY_THRESHOLD):
    """Best match for the face among `patient_id`'s people.
    Raises PatientNotFound when the patient has no folder."""
    folder = patient_folder(patient_id)
    if folder is None:
        raise PatientNotFound(patient_id)
    key = patient_key(patient_id)

    gallery = patient_gallery(key, folder)
    if not gallery:
        # Checked before the face model runs: nothing to compare with.
        return {
            "status": "empty",
            "patient_id": key,
            "message": f"Patient {key} has no registered people with "
                       "a usable photo.",
        }

    query, failure = face_query(read_image(image_path))
    if failure:
        return {**failure, "patient_id": key}

    scored = gallery.rank(query)
    top = [{"name": p["name"], "score": round(s, 4)} for s, p in scored]
    score, best = scored[0]

    if score >= threshold:
        return {
            "status": "match",
            "patient_id": key,
            "name": best["name"],
            "score": round(score, 4),
            "threshold": threshold,
            "sample_count": best["sample_count"],
            "top_matches": top,
        }

    return {
        "status": "unknown",
        "patient_id": key,
        "name": "Unknown",
        "score": round(score, 4),
        "threshold": threshold,
        "top_matches": top,
    }
