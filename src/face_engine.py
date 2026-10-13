from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .config import DETECTION_SIZE, MODEL_NAME


@lru_cache(maxsize=1)
def get_face_engine() -> FaceAnalysis:
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    return app


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def detect_faces(image: np.ndarray):
    # Tightly cropped small images (e.g. 160x160 dataset faces) fill the whole
    # detector input and get missed; a border gives the detector context.
    h, w = image.shape[:2]
    pad = max(h, w) // 2 if min(h, w) < 320 else 0
    if pad:
        image = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    faces = get_face_engine().get(image)

    # Map coordinates back to the original image.
    for face in faces:
        face.bbox = face.bbox - pad
        if face.get("kps") is not None:
            face.kps = face.kps - pad
    return faces


def normalized_embedding(face) -> np.ndarray:
    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Zero-length embedding")
    return embedding / norm
