from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from insightface import model_zoo
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils.storage import ensure_available

from .config import DETECTION_SIZE, DETECTOR_FILE, DETECTOR_PACK, MODEL_NAME


@lru_cache(maxsize=1)
def get_face_engine() -> FaceAnalysis:
    # Only detection + ArcFace are used; skipping the age/gender and
    # landmark models makes each request faster without changing embeddings.
    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    # Swap in the small detector from another pack (e.g. buffalo_s
    # det_500m), keeping MODEL_NAME's ArcFace so embeddings stay comparable.
    if DETECTOR_PACK:
        pack_dir = ensure_available("models", DETECTOR_PACK, root="~/.insightface")
        detector = model_zoo.get_model(
            str(Path(pack_dir) / DETECTOR_FILE),
            providers=["CPUExecutionProvider"])
        app.models["detection"] = detector
        app.det_model = detector
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    return app


def read_image(image_path) -> np.ndarray:
    # Already-decoded images (the API decodes uploads in memory) pass through.
    if isinstance(image_path, np.ndarray):
        return image_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def detect_faces(image: np.ndarray, embed=True):
    """embed=False runs only the detector; the ArcFace embedding is then
    computed later by normalized_embedding(face, image), for one face only."""
    # Tightly cropped small images (e.g. 160x160 dataset faces) fill the whole
    # detector input and get missed; a border gives the detector context.
    h, w = image.shape[:2]
    pad = max(h, w) // 2 if min(h, w) < 320 else 0
    if pad:
        image = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    engine = get_face_engine()
    if embed:
        faces = engine.get(image)
    else:
        bboxes, kpss = engine.det_model.detect(image, max_num=0, metric="default")
        faces = [
            Face(bbox=bboxes[i, :4], kps=None if kpss is None else kpss[i],
                 det_score=bboxes[i, 4])
            for i in range(bboxes.shape[0])
        ]

    # Map coordinates back to the original image.
    for face in faces:
        face.bbox = face.bbox - pad
        if face.get("kps") is not None:
            face.kps = face.kps - pad
    return faces


def normalized_embedding(face, image=None) -> np.ndarray:
    if face.embedding is None:
        # Alignment warps from the original image; areas outside it are black,
        # same as the detector's padding, so the embedding is unchanged.
        get_face_engine().models["recognition"].get(image, face)
    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Zero-length embedding")
    return embedding / norm
