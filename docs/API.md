# API Documentation — CognivaEyewear Celebrity Recognition

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
| `POST` | `/recognize` | Identify the celebrity in an uploaded face image |

---

## `GET /`

Returns a simple HTML page (`text/html`) with a file picker that calls `POST /recognize` and prints the JSON result. The full UI is the React app in `frontend/`.

---

## `GET /health`

Reports whether the server is running and whether the SQLite database can be queried.

**Response `200 OK`**

```json
{
  "status": "ok",
  "database_exists": true
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when the server responds |
| `database_exists` | boolean | `true` if the `celebrities` table exists and can be queried |

**Example**

```bash
curl http://127.0.0.1:8000/health
```

---

## `POST /recognize`

Detects the face in an uploaded image, computes its ArcFace embedding, and compares it to every celebrity centroid in the database using cosine similarity. The best score must be **greater than 0.80** to count as a match (`SIMILARITY_THRESHOLD` in `src/config.py`).

### Request

`Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Image containing exactly one face. The part's content type must start with `image/` (JPG, PNG and WEBP are supported). |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/recognize \
     -F "file=@test_images/virat_test.jpg"
```

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);
const res = await fetch("http://127.0.0.1:8000/recognize", { method: "POST", body: form });
const data = await res.json();
```

### Response `200 OK`

Every processed image returns HTTP 200. The `status` field gives the outcome:

| `status` | Meaning |
|---|---|
| `match` | One face was found and the best score is above the threshold |
| `unknown` | One face was found but no celebrity scored above the threshold |
| `no_face` | No face was detected |
| `multiple_faces` | More than one face was detected |
| `database_empty` | The database has no celebrities. Build it with `python -m src.generate_centroids` |

#### `match`

```json
{
  "status": "match",
  "name": "Zac Efron",
  "category": "Actor",
  "description": "American actor",
  "details": {
    "Category": "Actor",
    "Language": "English",
    "Description": "American actor"
  },
  "score": 0.8265,
  "threshold": 0.8,
  "sample_count": 49,
  "top_matches": [
    { "name": "Zac Efron", "score": 0.8265 },
    { "name": "Hrithik Roshan", "score": 0.2255 },
    { "name": "Virat Kohli", "score": 0.1815 },
    { "name": "Brad Pitt", "score": 0.1401 },
    { "name": "Ellen Degeneres", "score": 0.1312 }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"match"` |
| `name` | string | Matched celebrity name |
| `category` | string | Category from the celebrity metadata |
| `description` | string | Short description from the metadata |
| `details` | object | Every column of the celebrity's row in `metadata/celebrities.xlsx`. Keys vary by celebrity. |
| `score` | number | Cosine similarity to the best centroid, rounded to 4 decimals (range −1 to 1) |
| `threshold` | number | Threshold used for the decision |
| `sample_count` | integer | Number of images averaged into the matched centroid |
| `top_matches` | array | Up to 5 best candidates, `{name, score}`, highest score first |

#### `unknown`

```json
{
  "status": "unknown",
  "name": "Unknown",
  "score": 0.6837,
  "threshold": 0.8,
  "top_matches": [
    { "name": "Akshay Kumar", "score": 0.6837 },
    { "name": "Hrithik Roshan", "score": 0.1558 },
    { "name": "Lisa Kudrow", "score": 0.1384 },
    { "name": "Priyanka Chopra", "score": 0.1092 },
    { "name": "Billie Eilish", "score": 0.0995 }
  ]
}
```

`name` is always `"Unknown"`. `score` is the best score found, which was not above `threshold`.

#### `no_face`

```json
{
  "status": "no_face",
  "message": "No face detected."
}
```

#### `multiple_faces`

```json
{
  "status": "multiple_faces",
  "message": "2 faces detected. Please use one face."
}
```

#### `database_empty`

```json
{
  "status": "database_empty",
  "message": "No celebrity embeddings are stored in the database."
}
```

### Error responses

| HTTP code | When | Body |
|---|---|---|
| `400 Bad Request` | The uploaded part's content type is missing or not `image/*` | `{"detail": "Upload an image file."}` |
| `422 Unprocessable Entity` | The `file` form field is missing | FastAPI validation error (`{"detail": [{"type": "missing", "loc": ["body", "file"], ...}]}`) |
| `500 Internal Server Error` | The file says it is an image but cannot be decoded | `Internal Server Error` |

### Notes

- The first request after the server starts is slower because the InsightFace `buffalo_l` model is loaded then and cached for later requests.
- Uploaded images are written to a temporary file, processed, and deleted immediately. Nothing is stored.
- Scores come from L2-normalized 512-dimension ArcFace embeddings compared with L2-normalized per-celebrity centroids.
