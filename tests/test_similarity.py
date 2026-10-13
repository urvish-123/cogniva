from types import SimpleNamespace

import numpy as np

from src import recognize as recognize_module
from src.recognize import cosine_similarity, recognize_image


def test_identical_vectors():
    x = np.array([1, 2, 3], dtype=np.float32)
    assert abs(cosine_similarity(x, x) - 1.0) < 1e-6


def test_orthogonal_vectors():
    x = np.array([1, 0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    assert abs(cosine_similarity(x, y)) < 1e-6


def fake_pipeline(monkeypatch, query, faces=1):
    face = SimpleNamespace(embedding=np.asarray(query, dtype=np.float32))
    celebrities = [
        {"name": "Virat Kohli", "centroid": np.array([1, 0], np.float32),
         "sample_count": 35},
        {"name": "Brad Pitt", "centroid": np.array([0, 1], np.float32),
         "sample_count": 35},
    ]
    monkeypatch.setattr(recognize_module, "read_image", lambda path: None)
    monkeypatch.setattr(
        recognize_module, "detect_faces", lambda image: [face] * faces)
    monkeypatch.setattr(
        recognize_module, "get_all_celebrities", lambda: celebrities)


def test_match_above_threshold(monkeypatch):
    fake_pipeline(monkeypatch, [0.95, 0.1])
    result = recognize_image("x.jpg")
    assert result["status"] == "match"
    assert result["name"] == "Virat Kohli"
    assert result["score"] > 0.80


def test_unknown_below_threshold(monkeypatch):
    fake_pipeline(monkeypatch, [1, 1])  # ~0.707 to both centroids
    result = recognize_image("x.jpg")
    assert result["status"] == "unknown"
    assert result["name"] == "Unknown"


def test_multiple_faces(monkeypatch):
    fake_pipeline(monkeypatch, [1, 0], faces=2)
    assert recognize_image("x.jpg")["status"] == "multiple_faces"


def test_no_face(monkeypatch):
    fake_pipeline(monkeypatch, [1, 0], faces=0)
    assert recognize_image("x.jpg")["status"] == "no_face"
