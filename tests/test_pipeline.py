from src import pipeline


def fake_pipeline(monkeypatch, face_status, object_status="unknown"):
    calls = {"object": 0}

    def fake_face(path):
        return {"status": face_status, "name": "Virat Kohli"}

    def fake_object(path):
        calls["object"] += 1
        return {"status": object_status, "name": "Bed"}

    monkeypatch.setattr(pipeline, "recognize_image", fake_face)
    monkeypatch.setattr(pipeline, "recognize_object", fake_object)
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


def test_unknown_face_still_checks_objects(monkeypatch):
    calls = fake_pipeline(monkeypatch, "unknown", object_status="match")
    assert pipeline.recognize("x.jpg")["type"] == "object"
    assert calls["object"] == 1


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
