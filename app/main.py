from contextlib import asynccontextmanager
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from src.database import database_available, get_all_objects
from src.face_engine import get_face_engine
from src.object_engine import get_clip_model
from src.pipeline import recognize as recognize_pipeline

MAX_IMAGE_SIDE = 1280


def warm_up():
    # The first inference of each model is several times slower than later
    # ones; run one blank image (no face -> face + CLIP both run) at startup.
    recognize_pipeline(np.full((320, 320, 3), 128, dtype=np.uint8))


def decode_image(data: bytes):
    """Decode the upload once, in memory; both models share the array."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Models never use more than this (detector 480, CLIP 224); smaller
    # images make decoding, detection and CLIP preprocessing faster.
    scale = MAX_IMAGE_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return image


@asynccontextmanager
async def lifespan(app):
    # Load both models once at startup so request times exclude model loading.
    get_face_engine()
    get_clip_model()
    warm_up()
    yield


app = FastAPI(
    title="CognivaEyewear Celebrity & Object Recognition",
    version="2.0.0",
    lifespan=lifespan,
)

# React dev server (frontend/) runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cogniva Recognition</title>
<style>
body{font-family:Arial,sans-serif;max-width:760px;margin:50px auto;padding:24px}
.card{border:1px solid #ddd;border-radius:12px;padding:24px}
button{padding:10px 16px;margin-top:12px}
pre{background:#f6f6f6;padding:16px;border-radius:8px;white-space:pre-wrap}
</style>
</head>
<body>
<div class="card">
<h1>Cogniva Recognition</h1>
<p>Upload a celebrity photo or an object photo.</p>
<input id="file" type="file" accept="image/*">
<br>
<button onclick="recognize()">Recognize</button>
<pre id="result">Waiting for image...</pre>
</div>
<script>
async function recognize(){
  const f=document.getElementById('file').files[0];
  const out=document.getElementById('result');
  if(!f){out.textContent='Please select an image.';return;}
  const form=new FormData(); form.append('file',f);
  out.textContent='Processing...';
  const r=await fetch('/recognize',{method:'POST',body:form});
  const data=await r.json();
  out.textContent=JSON.stringify(data,null,2);
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database_exists": database_available(),
        "object_classes": len(get_all_objects()),
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    received = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail="Use a JPG, PNG or WEBP image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    image = decode_image(data)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    result = recognize_pipeline(image)

    # Upload validation + recognition, i.e. the full server-side time.
    result["processing_time"]["request_ms"] = round(
        (time.perf_counter() - received) * 1000, 1)
    return result
