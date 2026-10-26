from src import pipeline


def fake_pipeline(monkeypatch, face_status, object_status="unknown",
                  patient_status="match"):
    calls = {"face": 0, "object": 0, "patient": []}

    def fake_face(path):
        calls["face"] += 1
        return {"status": face_status, "name": "Virat Kohli"}

    def fake_object(path):
        calls["object"] += 1
        return {"status": object_status, "name": "Bed"}

    def fake_patient(path, patient_id):
        calls["patient"].append(patient_id)
        return {"status": patient_status, "name": "Son of amitab",
                "patient_id": "P001"}

    monkeypatch.setattr(pipeline, "recognize_image", fake_face)
    monkeypatch.setattr(pipeline, "recognize_object", fake_object)
    monkeypatch.setattr(pipeline, "recognize_patient", fake_patient)
    return calls


def test_celebrity_match_skips_object_model(monkeypatch):
    calls = fake_pipeline(monkeypatch, "match")
    result = pipeline.recognize("x.jpg")
    assert result["type"] == "celebrity"
    assert result["object"] is None
    assert calls["object"] == 0


def test_no_face_falls_back_to_object(monkeypatch):
    fake_pipeline(monkeypatch, "no_face", object_status="match")
    result = pipeline.recognize("x.jpg")
    assert result["type"] == "object"
    assert result["object"]["name"] == "Bed"


def test_unknown_face_is_not_recognized(monkeypatch):
    calls = fake_pipeline(monkeypatch, "unknown", object_status="match")
    assert pipeline.recognize("x.jpg")["type"] == "unknown"
    assert calls["object"] == 0


def test_nothing_matches(monkeypatch):
    fake_pipeline(monkeypatch, "no_face", object_status="unknown")
    assert pipeline.recognize("x.jpg")["type"] == "unknown"


def test_multiple_faces_is_not_sent_to_objects(monkeypatch):
    calls = fake_pipeline(monkeypatch, "multiple_faces")
    result = pipeline.recognize("x.jpg")
    assert result["type"] == "unknown"
    assert result["celebrity"]["status"] == "multiple_faces"
    assert calls["object"] == 0


def test_processing_time_is_reported(monkeypatch):
    fake_pipeline(monkeypatch, "no_face")
    timings = pipeline.recognize("x.jpg")["processing_time"]
    assert {"face_ms", "object_ms", "total_ms"} <= set(timings)
    assert timings["total_ms"] >= timings["face_ms"]



def test_without_patient_id_patients_are_not_searched(monkeypatch):
    calls = fake_pipeline(monkeypatch, "match")
    result = pipeline.recognize("x.jpg")
    assert result["type"] == "celebrity"
    assert result["patient"] is None
    assert calls["patient"] == []


def test_patient_match_searches_only_that_patient(monkeypatch):
    calls = fake_pipeline(monkeypatch, "match", object_status="match")
    result = pipeline.recognize("x.jpg", patient_id="P001")
    assert result["type"] == "patient"
    assert result["patient"]["name"] == "Son of amitab"
    assert result["celebrity"] is None and result["object"] is None
    assert calls == {"face": 0, "object": 0, "patient": ["P001"]}
    assert set(result["processing_time"]) == {"face_ms", "total_ms"}


def test_patient_miss_does_not_fall_back(monkeypatch):
    # Celebrity and object would both match; with a patient_id neither runs.
    for status in ("unknown", "no_face", "multiple_faces", "empty"):
        calls = fake_pipeline(monkeypatch, "match", object_status="match",
                              patient_status=status)
        result = pipeline.recognize("x.jpg", patient_id="P001")
        assert result["type"] == "unknown"
        assert result["patient"]["status"] == status
        assert calls["face"] == 0 and calls["object"] == 0
