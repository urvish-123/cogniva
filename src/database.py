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


@contextmanager
def connection():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
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
