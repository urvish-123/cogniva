import json
import sqlite3
from contextlib import contextmanager

import numpy as np

from .config import DATABASE_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS celebrities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    language TEXT,
    description TEXT,
    centroid_embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details TEXT NOT NULL DEFAULT '{}'
)
"""

# CLIP object centroids. Separate table: never compared with ArcFace vectors.
OBJECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    centroid_embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


# ArcFace centroids of people registered by a patient: one row per person
# folder (dataset/Patient/<patient_id>/<folder>). folder_mtime is the
# folder's modification time when it was embedded; a different time means
# photos were added or removed, and only that person is embedded again.
PATIENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    folder TEXT NOT NULL,
    centroid_embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    folder_mtime REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patient_id, folder)
)
"""


# Read results kept in memory, keyed by query, together with the database
# file's modification time: requests skip SQLite until the file changes
# (e.g. a centroid rebuild in another process). Writes here clear it too.
_cache = {}


def _cached(key, load):
    try:
        version = DATABASE_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        version = None
    hit = _cache.get(key)
    if hit is not None and hit[0] == version:
        return hit[1]
    value = load()
    _cache[key] = (version, value)
    return value


@contextmanager
def connection():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
        if conn.total_changes:
            _cache.clear()
        conn.commit()
    finally:
        conn.close()


def init_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.execute(SCHEMA)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_celebrities_name ON celebrities(name)"
        )
        conn.execute(OBJECT_SCHEMA)
        # Renamed to patient_people; rebuilt from dataset/Patient.
        conn.execute("DROP TABLE IF EXISTS customer_people")
        columns = {row[1] for row in
                   conn.execute("PRAGMA table_info(patient_people)")}
        if columns and "folder_mtime" not in columns:
            # Older layout; the table is only an index of the patient
            # folders, so it is dropped and rebuilt from the photos.
            conn.execute("DROP TABLE patient_people")
        conn.execute(PATIENT_SCHEMA)


def database_available():
    try:
        with connection() as conn:
            conn.execute("SELECT COUNT(*) FROM celebrities")
            return True
    except sqlite3.Error:
        return False


def clear_celebrities():
    with connection() as conn:
        conn.execute("DELETE FROM celebrities")


def upsert_celebrity(
    name,
    centroid,
    sample_count,
    category="",
    language="",
    description="",
    details=None,
):
    centroid = np.asarray(centroid, dtype=np.float32)
    blob = centroid.tobytes()
    details_json = json.dumps(details or {}, ensure_ascii=False)

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO celebrities
              (name, category, language, description, details,
               centroid_embedding, embedding_dimension, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              category=excluded.category,
              language=excluded.language,
              description=excluded.description,
              details=excluded.details,
              centroid_embedding=excluded.centroid_embedding,
              embedding_dimension=excluded.embedding_dimension,
              sample_count=excluded.sample_count
            """,
            (
                name, category, language, description, details_json, blob,
                int(centroid.shape[0]), int(sample_count)
            ),
        )


def get_all_celebrities():
    return _cached("celebrities", _load_celebrities)


def _load_celebrities():
    init_database()
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, category, language, description, details,
                   centroid_embedding, embedding_dimension, sample_count
            FROM celebrities
            ORDER BY name
            """
        ).fetchall()

    records = []
    for row in rows:
        records.append({
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "language": row[3],
            "description": row[4],
            "details": json.loads(row[5] or "{}"),
            "centroid": np.frombuffer(
                row[6], dtype=np.float32, count=row[7]
            ),
            "sample_count": row[8],
        })
    return records


def clear_objects():
    init_database()
    with connection() as conn:
        conn.execute("DELETE FROM objects")


def upsert_object(name, centroid, sample_count, category=""):
    centroid = np.asarray(centroid, dtype=np.float32)

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO objects
              (name, category, centroid_embedding, embedding_dimension,
               sample_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              category=excluded.category,
              centroid_embedding=excluded.centroid_embedding,
              embedding_dimension=excluded.embedding_dimension,
              sample_count=excluded.sample_count
            """,
            (name, category, centroid.tobytes(), int(centroid.shape[0]),
             int(sample_count)),
        )


def get_all_objects():
    return _cached("objects", _load_objects)


def _load_objects():
    init_database()
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, category, centroid_embedding,
                   embedding_dimension, sample_count
            FROM objects
            ORDER BY name
            """
        ).fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "centroid": np.frombuffer(row[3], dtype=np.float32, count=row[4]),
            "sample_count": row[5],
        }
        for row in rows
    ]


def delete_patient_people(patient_id=None, folders=None):
    """Remove people: every patient's (no arguments), one patient's, or
    only the given person folders of one patient."""
    init_database()
    with connection() as conn:
        if patient_id is None:
            conn.execute("DELETE FROM patient_people")
        elif folders is None:
            conn.execute("DELETE FROM patient_people WHERE patient_id = ?",
                         (patient_id,))
        else:
            conn.executemany(
                "DELETE FROM patient_people"
                " WHERE patient_id = ? AND folder = ?",
                [(patient_id, folder) for folder in folders])


def upsert_patient_person(patient_id, folder, centroid, sample_count,
                          folder_mtime):
    centroid = np.asarray(centroid, dtype=np.float32)

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_people
              (patient_id, folder, centroid_embedding, embedding_dimension,
               sample_count, folder_mtime)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id, folder) DO UPDATE SET
              centroid_embedding=excluded.centroid_embedding,
              embedding_dimension=excluded.embedding_dimension,
              sample_count=excluded.sample_count,
              folder_mtime=excluded.folder_mtime,
              created_at=CURRENT_TIMESTAMP
            """,
            (patient_id, folder, centroid.tobytes(), int(centroid.shape[0]),
             int(sample_count), float(folder_mtime)),
        )


def get_patient_people(patient_id):
    """One patient's indexed people. `name` is the folder name for display
    ("Son_of_amitab" -> "Son of amitab")."""
    return _cached(("patient_people", patient_id),
                   lambda: _load_patient_people(patient_id))


def _load_patient_people(patient_id):
    init_database()
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT folder, centroid_embedding, embedding_dimension,
                   sample_count, folder_mtime
            FROM patient_people
            WHERE patient_id = ?
            ORDER BY folder
            """,
            (patient_id,),
        ).fetchall()

    return [
        {
            "folder": row[0],
            "name": row[0].replace("_", " "),
            "centroid": np.frombuffer(row[1], dtype=np.float32, count=row[2]),
            "sample_count": row[3],
            "folder_mtime": row[4],
        }
        for row in rows
    ]
