import os

import numpy as np
import pytest

from src import patient, database

# Fake "embedding" per person folder; the query below is closest to SON.
SON = np.array([1, 0, 0], np.float32)
WIFE = np.array([0, 1, 0], np.float32)
CENTROIDS = {"Son_of_amitab": SON, "Wife_of_amitab": WIFE}


@pytest.fixture
def patients(monkeypatch, tmp_path):
    """dataset/Patient with P001 (two people), a temporary database,
    and fake face models. Returns the list of embedded person folders."""
    root = tmp_path / "Patient"
    for person in CENTROIDS:
        (root / "P001" / person).mkdir(parents=True)
        (root / "P001" / person / "1.jpg").write_bytes(b"")
    (root / "P002").mkdir()

    monkeypatch.setattr(patient, "PATIENT_DATASET_DIR", root)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(database, "_cache", {})
    monkeypatch.setattr(patient, "_galleries", {})

    embedded = []

    def fake_centroid(images):
        folder = images[0].parent.name
        embedded.append(folder)
        return CENTROIDS.get(folder, SON), len(images), 0

    monkeypatch.setattr(patient, "folder_centroid", fake_centroid)
    monkeypatch.setattr(patient, "read_image", lambda path: path)
    monkeypatch.setattr(patient, "face_query",
                        lambda query: (np.asarray(query, np.float32), None))
    return root, embedded


def test_best_match_within_the_patient(patients):
    result = patient.recognize_patient([0.95, 0.2, 0], "P001")
    assert result["status"] == "match"
    assert result["name"] == "Son of amitab"
    assert result["patient_id"] == "P001"
    assert [m["name"] for m in result["top_matches"]] == [
        "Son of amitab", "Wife of amitab"]


def test_no_match_below_threshold(patients):
    result = patient.recognize_patient([0.3, 0.3, 1], "P001")  # ~0.28
    assert result["status"] == "unknown"
    assert result["name"] == "Unknown"
    assert result["top_matches"]


def test_patient_id_is_case_insensitive(patients):
    assert patient.recognize_patient([1, 0, 0], " p001 ")["patient_id"] == "P001"


@pytest.mark.parametrize("patient_id", ["P999", "../P001", "P001/..", ""])
def test_unknown_or_invalid_patient(patients, patient_id):
    with pytest.raises(patient.PatientNotFound):
        patient.recognize_patient([1, 0, 0], patient_id)


def test_patient_without_people_skips_face_model(patients, monkeypatch):
    monkeypatch.setattr(patient, "face_query", lambda image: pytest.fail())
    assert patient.recognize_patient([1, 0, 0], "P002")["status"] == "empty"


def test_face_problems_are_reported(patients, monkeypatch):
    monkeypatch.setattr(patient, "face_query",
                        lambda image: (None, {"status": "no_face"}))
    result = patient.recognize_patient([1, 0, 0], "P001")
    assert result == {"status": "no_face", "patient_id": "P001"}


def test_unchanged_patient_is_served_from_memory(patients, monkeypatch):
    _, embedded = patients
    patient.recognize_patient([1, 0, 0], "P001")
    assert sorted(embedded) == ["Son_of_amitab", "Wife_of_amitab"]

    monkeypatch.setattr(patient, "get_patient_people",
                        lambda key: pytest.fail("database read"))
    patient.recognize_patient([1, 0, 0], "P001")
    assert len(embedded) == 2


def test_index_survives_restart_without_re_embedding(patients):
    _, embedded = patients
    patient.recognize_patient([1, 0, 0], "P001")
    patient._galleries.clear()  # new process, same database
    patient.recognize_patient([1, 0, 0], "P001")
    assert len(embedded) == 2


def test_only_changed_people_are_embedded_again(patients):
    root, embedded = patients
    patient.recognize_patient([1, 0, 0], "P001")
    embedded.clear()

    # New person, and a removed one.
    (root / "P001" / "Daughter").mkdir()
    (root / "P001" / "Daughter" / "1.jpg").write_bytes(b"")
    for photo in (root / "P001" / "Wife_of_amitab").iterdir():
        photo.unlink()
    (root / "P001" / "Wife_of_amitab").rmdir()

    result = patient.recognize_patient([1, 0, 0], "P001")
    assert embedded == ["Daughter"]
    assert {m["name"] for m in result["top_matches"]} == {
        "Son of amitab", "Daughter"}


def test_added_photo_re_embeds_that_person(patients):
    root, embedded = patients
    patient.recognize_patient([1, 0, 0], "P001")
    embedded.clear()

    folder = root / "P001" / "Son_of_amitab"
    (folder / "2.jpg").write_bytes(b"")
    # Folder times can be coarse; make sure this one visibly changed.
    os.utime(folder, (folder.stat().st_atime, folder.stat().st_mtime + 5))

    patient.recognize_patient([1, 0, 0], "P001")
    assert embedded == ["Son_of_amitab"]
