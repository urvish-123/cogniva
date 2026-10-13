# CognivaEyewear Celebrity Recognition — Final Project

## What this project does

This is a data and embedding engineering implementation for an offline-oriented celebrity recognition prototype.

It does **not** train a new face-recognition model. It uses a pretrained InsightFace/ArcFace pipeline to create normalized face embeddings, averages the embeddings per identity into one centroid, stores the centroid plus metadata in SQLite (`database/celebrity_embeddings.db`), and recognizes new faces using cosine similarity.

## Technology

- Python
- InsightFace / ArcFace
- RetinaFace-compatible detection through InsightFace
- OpenCV
- NumPy
- SQLite (Python standard library)
- FastAPI
- React + Vite (web frontend in `frontend/`)
- KaggleHub for dataset acquisition

## Documentation

- API reference: [docs/API.md](docs/API.md) (also available live at `http://127.0.0.1:8000/docs` while the server is running)
- Database schema script: [database/schema.sql](database/schema.sql)
- Full database dump (schema + data): [database/celebrity_embeddings_dump.sql](database/celebrity_embeddings_dump.sql)

## End-to-end workflow

```text
Kaggle dataset
    ↓
prepare_kaggle_dataset.py
    ↓
celebrity_dataset/
    ↓
preprocess.py
    ↓
cleaned_dataset/
    ↓
InsightFace / ArcFace
    ↓
image embeddings (memory only)
    ↓
mean + normalization
    ↓
one centroid per celebrity
    ↓
SQLite database/celebrity_embeddings.db
    ↓
new test image
    ↓
ArcFace embedding
    ↓
cosine similarity
    ↓
score > 0.80 → match
score <= 0.80 → Unknown
```

## 1. Create environment

Python 3.10 or 3.11 is recommended.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Database

No database server is needed. Embeddings are stored in the SQLite file `database/celebrity_embeddings.db`, which is created automatically the first time it is needed. Set `CELEBRITY_DB_PATH` to use a different file.

Database scripts:

| File | Use |
|---|---|
| `database/schema.sql` | Creates the empty `celebrities` table and index |
| `database/celebrity_embeddings_dump.sql` | Recreates the full database (31 celebrities) without rebuilding embeddings |

Restore the full database without re-running the pipeline:

```bash
python -c "import sqlite3; sqlite3.connect('database/celebrity_embeddings.db').executescript(open('database/celebrity_embeddings_dump.sql', encoding='utf-8').read())"
```

(or `sqlite3 database/celebrity_embeddings.db < database/celebrity_embeddings_dump.sql` if the SQLite CLI is installed). Delete any existing `.db` file first.

## 2. Download the requested Kaggle dataset

```bash
python -m src.download_kaggle_dataset
```

The downloader uses:

```text
cybersimar08/face-recognition-dataset
```

Kaggle authentication must be configured as required by kagglehub on your machine.

## 3. Prepare the dataset

Use the exact local path printed by the previous command:

```bash
python -m src.prepare_kaggle_dataset --source "PASTE_KAGGLEHUB_PATH_HERE"
```

Default behavior:
- recursively finds image-containing identity folders
- copies up to 50 images per identity
- uses safe ASCII output names
- writes `metadata/kaggle_selection.csv`

Review this result carefully. The script discovers folder names automatically; it does not assume every identity in the source dataset is a celebrity.

## 4. Clean the images

```bash
python -m src.preprocess
```

This creates:

```text
cleaned_dataset/
metadata/dataset_quality_report.csv
```

Images are rejected when:
- unreadable
- too small
- zero/multiple faces
- face is too small relative to image
- blur score is below the configured threshold

## 5. Build the SQLite embedding database

```bash
python -m src.generate_centroids --dataset cleaned_dataset
```

Output: the `celebrities` table in `database/celebrity_embeddings.db`.

### Database schema

```sql
CREATE TABLE celebrities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    language TEXT,
    description TEXT,
    centroid_embedding BLOB NOT NULL,      -- float32 vector
    embedding_dimension INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details TEXT NOT NULL DEFAULT '{}'     -- JSON
);
```

Only the final centroid is persisted; individual image embeddings are held in memory during generation.

### Training and test images

Every clean image of a celebrity is used to build their centroid. Every image in `celebrity_dataset/` (for celebrities in the metadata) is also used as a test case, read in place. Because these images are also used for training, the accuracy shows how well the database fits the dataset, not how well it handles new photos.

Only people listed in `metadata/celebrities.xlsx` are stored in the database. Folders for anyone else are skipped.

Optional: an `unknown_people/` folder (one subfolder per person) can hold photos of people who are not in the Excel file. They are never stored in the database. Two single-face photos of each are used as "Unknown person" test cases. This folder is not included in the current submission, so `test_cases.csv` currently has no "Unknown person" rows.

The same command also writes `metadata/test_cases.csv`:

| Test | Image(s) | Expected |
|---|---|---|
| Celebrity dataset image | every image in `celebrity_dataset/<name>/` | Correct celebrity |
| Different lighting | `lighting_dark_test.jpg` (darkened dataset image) | Correct celebrity |
| Unknown person | `unknown_<name>_XX.jpg` | Unknown |
| Multiple faces | `multiple_faces_test.jpg` (two dataset faces side by side) | multiple_faces |

### Celebrity details from Excel

`metadata/celebrities.xlsx` (150 Indian TV celebrities) holds the celebrity details. Every column in its first sheet is stored in the `details` column of the database and shown in the app after a match. The name column must be called `Celebrity` or `Name`. Names are matched to photo folders ignoring case, spaces and punctuation, so the `P_V_Sindhu` folder matches "P. V. Sindhu".

To add photos for a celebrity:

1. Put 20–50 clear photos (one face each) in their folder, e.g. `celebrity_dataset/Salman_Khan/`. An empty folder already exists for every celebrity in the Excel file.
2. Re-run cleaning and the database build:
   ```bash
   python -m src.preprocess
   python -m src.generate_centroids --dataset cleaned_dataset
   ```
3. Check who is ready and who still needs photos:
   ```bash
   python -m src.metadata
   ```
   This writes `metadata/coverage_report.csv`. A celebrity needs at least 10 clean images (`MIN_IMAGES_PER_IDENTITY`). Anyone with fewer is skipped.

## 6. Test one image

Use any image (for example your own `test_images/virat_test.jpg`).

```bash
python -m src.recognize test_images/virat_test.jpg
```

Possible results:

```text
match → celebrity name + score
unknown → Unknown + score
no_face → no face detected
multiple_faces → input contains multiple faces
database_empty → database must be built first
```

## 7. Run the web app

From the project root:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the minimal built-in page, or `http://127.0.0.1:8000/docs` for interactive API docs. Endpoints are described in [docs/API.md](docs/API.md).

### React frontend

Requires Node.js 18+. With the backend running, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite forwards `/recognize` and `/health` to the backend on port 8000. `npm run build` writes a production build to `frontend/dist/`.

## 8. Run automated tests

```bash
pytest
```

## 9. Evaluate the test set

`metadata/test_cases.csv` is generated in step 5. You can add your own rows:

```csv
image,expected_name,case
test_images/virat_test.jpg,Virat Kohli,Same person, different photo
test_images/stranger.jpg,Unknown,Unknown person
```

Run:

```bash
python -m src.evaluate --csv metadata/test_cases.csv
```

It prints PASS/FAIL for each image, a score for each test type, and the overall accuracy. Only use images that were not used to build the centroids.

## Final deliverables

Submit:

```text
Cogniva_Celebrity_Recognition_Final/
├── app/                  FastAPI backend
├── src/                  data pipeline + recognition
├── frontend/             React web app
├── tests/
├── database/
│   ├── celebrity_embeddings.db
│   ├── schema.sql
│   └── celebrity_embeddings_dump.sql
├── docs/
│   └── API.md
├── celebrity_dataset/
├── cleaned_dataset/
├── metadata/
│   ├── celebrities.xlsx
│   ├── celebrities.csv
│   ├── kaggle_selection.csv
│   ├── dataset_quality_report.csv
│   └── test_cases.csv
├── test_images/
├── requirements.txt
└── README.md
```

