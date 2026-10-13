from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.database import database_available
from src.recognize import recognize_image

app = FastAPI(
    title="CognivaEyewear Celebrity Recognition",
    version="1.0.0",
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
<title>Cogniva Celebrity Recognition</title>
<style>
body{font-family:Arial,sans-serif;max-width:760px;margin:50px auto;padding:24px}
.card{border:1px solid #ddd;border-radius:12px;padding:24px}
button{padding:10px 16px;margin-top:12px}
pre{background:#f6f6f6;padding:16px;border-radius:8px;white-space:pre-wrap}
</style>
</head>
<body>
<div class="card">
<h1>Cogniva Celebrity Recognition</h1>
<p>Upload one clear face image.</p>
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
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = Path(temp.name)

    try:
        return recognize_image(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
