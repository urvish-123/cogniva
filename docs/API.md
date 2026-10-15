# API Documentation — CognivaEyewear Celebrity & Object Recognition

FastAPI backend defined in `app/main.py`.

| | |
|---|---|
| Base URL (local) | `http://127.0.0.1:8000` |
| Start server | `uvicorn app.main:app --reload` (from the project root) |
| Authentication | None |
| Interactive docs | `http://127.0.0.1:8000/docs` (Swagger UI), `/redoc` |
| OpenAPI schema | `http://127.0.0.1:8000/openapi.json` |
| CORS | Allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173` (React dev server) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Minimal built-in HTML upload page |
| `GET` | `/health` | Service and database status |
| `POST` | `/recognize` | Identify the celebrity or object in an uploaded image |

---

## `GET /`

Returns a simple HTML page (`text/html`) with a file picker that calls `POST /recognize` and prints the JSON result. The full UI is the React app in `frontend/`.

---

## `GET /health`

**Response `200 OK`**

```json
{
  "status": "ok",
  "database_exists": true,
  "object_classes": 8
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when the server responds |
| `database_exists` | boolean | `true` if the `celebrities` table exists and can be queried |
| `object_classes` | integer | Number of object centroids in the `objects` table |

---

## `POST /recognize`

Two specialised models are used, and their embeddings are never compared with each other:

| Step | Model | Compared against |
|---|---|---|
| 1. Face | InsightFace `buffalo_l` (RetinaFace detection + ArcFace) | `celebrities` table |
| 2. Object | CLIP `openai/clip-vit-base-patch32` | `objects` table |

Decision logic (`src/pipeline.py`):

1. Run face recognition (unchanged celebrity logic, threshold `SIMILARITY_THRESHOLD`).
   - Celebrity match → `type: "celebrity"`, CLIP is skipped.
   - More than one face → `type: "unknown"` (one face per image, as before).
2. No face, or a face that matches no celebrity → run CLIP object recognition (threshold `OBJECT_SIMILARITY_THRESHOLD`).
   - Object match → `type: "object"`.
   - Otherwise → `type: "unknown"`.

Both thresholds are in `src/config.py`. A score must be **greater than** the threshold to count as a match.

### Request

`Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | JPG, PNG or WEBP image, at most 10 MB |

```bash
curl -X POST http://127.0.0.1:8000/recognize -F "file=@test_images/virat_test.jpg"
```

### Response `200 OK`

| Field | Type | Description |
|---|---|---|
| `type` | string | `"celebrity"`, `"object"` or `"unknown"` |
| `celebrity` | object | Face recognition result (always present). Same fields as before: `status` is `match`, `unknown`, `no_face`, `multiple_faces` or `database_empty` |
| `object` | object or null | CLIP result, or `null` when CLIP was skipped. `status` is `match`, `unknown` or `database_empty` |
| `processing_time` | object | Milliseconds: `face_ms`, `object_ms` (only if CLIP ran), `total_ms` (recognition), `request_ms` (validation + recognition) |

#### Celebrity image

```json
{
  "type": "celebrity",
  "celebrity": {
    "status": "match",
    "name": "Zac Efron",
    "category": "Actor",
    "description": "American actor",
    "details": { "Category": "Actor", "Language": "English" },
    "score": 0.8265,
    "threshold": 0.8,
    "sample_count": 49,
    "top_matches": [{ "name": "Zac Efron", "score": 0.8265 }]
  },
  "object": null,
  "processing_time": { "face_ms": 142.3, "total_ms": 142.4, "request_ms": 150.9 }
}
```

#### Object image

```json
{
  "type": "object",
  "celebrity": { "status": "no_face", "message": "No face detected." },
  "object": {
    "status": "match",
    "name": "Bed",
    "category": "Furniture",
    "score": 0.8731,
    "threshold": 0.8,
    "sample_count": 27,
    "top_matches": [{ "name": "Bed", "score": 0.8731 }, { "name": "Armchair", "score": 0.7412 }]
  },
  "processing_time": { "face_ms": 98.1, "object_ms": 187.6, "total_ms": 285.8, "request_ms": 291.2 }
}
```

#### Not recognized

```json
{
  "type": "unknown",
  "celebrity": { "status": "no_face", "message": "No face detected." },
  "object": { "status": "unknown", "name": "Unknown", "score": 0.61, "threshold": 0.8, "top_matches": [] },
  "processing_time": { "face_ms": 95.0, "object_ms": 180.2, "total_ms": 275.3, "request_ms": 280.0 }
}
```

### Error responses

| HTTP code | When | Body |
|---|---|---|
| `400 Bad Request` | Not an image, unsupported extension, empty file, or the image cannot be decoded | `{"detail": "..."}` |
| `413 Payload Too Large` | File is larger than 10 MB | `{"detail": "Image must be 10 MB or smaller."}` |
| `422 Unprocessable Entity` | The `file` form field is missing | FastAPI validation error |

### Notes

- Both models are loaded once at server startup, so `processing_time` never includes model loading. Startup takes longer instead (the first run also downloads CLIP from Hugging Face).
- Uploaded images are written to a temporary file, processed, and deleted immediately. Nothing is stored.
