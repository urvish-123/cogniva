# API Documentation — CognivaEyewear Celebrity & Object Recognition

FastAPI backend defined in `app/main.py`.

| | |
|---|---|
| Base URL (local) | `http://127.0.0.1:8000` |
| Start server | `uvicorn app.main:app --reload` (from the project root) |
| Authentication | None |
| Interactive docs | `http.0.0.1:8000/docs://127` (Swagger UI), `/redoc` |
| OpenAPI schema | `http://127.0.0.1:8000/openapi.json` |
| CORS | Allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173` (React dev server) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Minimal built-in HTML upload page |
| `GET` | `/health` | Service and database status |
| `POST` | `/recognize` | Identify the celebrity or object in an uploaded image; with `patient_id`, identify which of that patient's people it shows |

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
| Face, patient (only with `patient_id`) | InsightFace `buffalo_l` (ArcFace) | `patient_people` rows of that patient only |
| Face, celebrity | InsightFace `buffalo_l` (RetinaFace detection + ArcFace) | `celebrities` table |
| Object | CLIP `openai/clip-vit-base-patch32` | `objects` table |

Decision logic (`src/pipeline.py`):

**A. With `patient_id`** (patient flow; folder `dataset/Patient/<patient_id>`, case-insensitive):

- The face is compared with that patient's people **only**. Celebrities and CLIP never run.
- Best match ≥ `PATIENT_SIMILARITY_THRESHOLD` (0.45) → `type: "patient"` with the person's name (folder name, `_` shown as spaces).
- No match, no face, more than one face, or no registered people → `type: "unknown"`; `patient.status` says which.
- No folder for that patient → `404 Not Found`.

**B. Without `patient_id`** (general flow):

1. Run face recognition.
   - Celebrity best match ≥ `SIMILARITY_THRESHOLD` (0.80) → `type: "celebrity"`, CLIP is skipped.
   - No match → `type: "unknown"` (not recognized), CLIP is skipped.
   - More than one face → `type: "unknown"` (one face per image, as before).
2. No face → run CLIP object recognition (threshold `OBJECT_SIMILARITY_THRESHOLD`).
   - Object match → `type: "object"`.
   - Otherwise → `type: "unknown"`.

All thresholds are in `src/config.py`. Face matches count when the score is **greater than or equal to** the threshold; objects when it is **greater than** it.

### Request

`POST /recognize?patient_id=P001` — `Content-Type: multipart/form-data`

| Field | In | Type | Required | Description |
|---|---|---|---|---|
| `patient_id` | query | string | No | e.g. `P001`. Letters, digits, `_`, `-`; case-insensitive. Empty = not given. A `patient_id` form field is also accepted; the query parameter wins when both are sent |
| `file` | form | file | Yes | JPG, PNG or WEBP image, at most 10 MB |

```bash
curl -X POST http://127.0.0.1:8000/recognize -F "file=@test_images/virat_test.jpg"
curl -X POST "http://127.0.0.1:8000/recognize?patient_id=P001" -F "file=@photo.jpg"
```

Patient people come from `dataset/Patient/<patient_id>/<Person>/*.jpg`. Every patient is indexed at server startup and kept in memory. A request only lists the patient's person folders to detect changes; a person folder whose photos were added, removed or renamed is re-embedded on the next request (a few hundred ms per photo, once), nothing else. `python -m src.generate_patient_centroids [--patient P001] [--rebuild]` builds the index ahead of time.

### Response `200 OK`

| Field | Type | Description |
|---|---|---|
| `type` | string | `"patient"`, `"celebrity"`, `"object"` or `"unknown"` |
| `patient` | object or null | Patient flow only, else `null`. `status` is `match` (with `name`, `score`, `sample_count`), `unknown`, `no_face`, `multiple_faces` or `empty` (no registered people with a usable photo). Always has `patient_id`; `match`/`unknown` also have `threshold` and `top_matches` |
| `celebrity` | object or null | General flow only, else `null`. `status` is `match`, `unknown`, `no_face`, `multiple_faces` or `database_empty` |
| `object` | object or null | CLIP result, or `null` when CLIP did not run. `status` is `match`, `unknown` or `database_empty` |
| `processing_time` | object | Milliseconds: `face_ms`, `object_ms` (only if CLIP ran), `total_ms` (recognition), `request_ms` (validation + recognition) |

#### Patient image (`patient_id=P001`)

```json
{
  "type": "patient",
  "patient": {
    "status": "match",
    "patient_id": "P001",
    "name": "Son of amitab",
    "score": 0.7612,
    "threshold": 0.45,
    "sample_count": 8,
    "top_matches": [{ "name": "Son of amitab", "score": 0.7612 }, { "name": "Wife of amitab", "score": 0.0531 }]
  },
  "celebrity": null,
  "object": null,
  "processing_time": { "face_ms": 62.3, "total_ms": 62.3, "request_ms": 65.2 }
}
```

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
| `404 Not Found` | `patient_id` given but there is no folder `dataset/Patient/<patient_id>` | `{"detail": "Patient P999 not found."}` |
| `413 Payload Too Large` | File is larger than 10 MB | `{"detail": "Image must be 10 MB or smaller."}` |
| `422 Unprocessable Entity` | The `file` form field is missing | FastAPI validation error |

### Notes

- Both models are loaded once at server startup, so `processing_time` never includes model loading. Startup takes longer instead (the first run also downloads CLIP from Hugging Face).
- Uploaded images are written to a temporary file, processed, and deleted immediately. Nothing is stored.
