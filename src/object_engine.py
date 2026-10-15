from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CLIP_MODEL_NAME, CLIP_ONNX_PATH


def export_clip_onnx():
    """One-time export of the CLIP image model (+ projection) to ONNX."""
    # Imported lazily: torch is only needed for this export.
    import torch
    from transformers import CLIPModel

    # The repo ships pytorch_model.bin; use_safetensors=False stops
    # transformers re-downloading a ~600 MB safetensors copy.
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME, use_safetensors=False).eval()

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            features = self.model.get_image_features(pixel_values=pixel_values)
            # Newer Transformers versions return a model output instead of a tensor.
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            return features / features.norm(dim=-1, keepdim=True)

    CLIP_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CLIP_ONNX_PATH.with_suffix(".tmp")
    torch.onnx.export(
        ImageEncoder(), (torch.zeros(1, 3, 224, 224),), str(temp_path),
        input_names=["pixel_values"], output_names=["embedding"],
        opset_version=17, dynamo=False,
    )
    temp_path.replace(CLIP_ONNX_PATH)


@lru_cache(maxsize=1)
def get_clip_model():
    import onnxruntime as ort
    from transformers import CLIPImageProcessor

    if not CLIP_ONNX_PATH.exists():
        export_clip_onnx()

    session = ort.InferenceSession(
        str(CLIP_ONNX_PATH), providers=["CPUExecutionProvider"])
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    return session, processor


def image_embedding(image_path) -> np.ndarray:
    """L2-normalized 512-d CLIP image embedding (same as the object project).

    Accepts a file path or an already-decoded BGR image (np.ndarray)."""
    from PIL import Image

    session, processor = get_clip_model()

    if isinstance(image_path, np.ndarray):
        rgb = Image.fromarray(np.ascontiguousarray(image_path[:, :, ::-1]))
        pixels = processor(images=rgb, return_tensors="np")["pixel_values"]
    else:
        with Image.open(image_path) as image:
            pixels = processor(
                images=image.convert("RGB"), return_tensors="np")["pixel_values"]

    embedding = session.run(None, {"pixel_values": pixels.astype(np.float32)})[0][0]
    return embedding.astype(np.float32)
