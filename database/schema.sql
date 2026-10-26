-- CognivaEyewear Celebrity Recognition — SQLite schema
--
-- Database file: database/celebrity_embeddings.db (override with CELEBRITY_DB_PATH)
-- The app creates this schema automatically (src/database.py: init_database).
-- To create an empty database by hand:
--
--     sqlite3 database/celebrity_embeddings.db < database/schema.sql
--
-- To restore the full database (schema + 31 celebrity centroids) instead,
-- load database/celebrity_embeddings_dump.sql the same way.

CREATE TABLE IF NOT EXISTS celebrities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL UNIQUE,              -- display name from metadata/celebrities.xlsx
    category            TEXT,                                 -- e.g. "Actor / Producer"
    language            TEXT,                                 -- e.g. "Hindi"
    description         TEXT,                                 -- one-line description
    centroid_embedding  BLOB    NOT NULL,                     -- L2-normalized float32 vector (ArcFace, little-endian)
    embedding_dimension INTEGER NOT NULL,                     -- vector length; 512 for buffalo_l
    sample_count        INTEGER NOT NULL,                     -- number of face images averaged into the centroid
    created_at          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details             TEXT    NOT NULL DEFAULT '{}'         -- JSON: every column of the celebrity's Excel row
);

CREATE INDEX IF NOT EXISTS idx_celebrities_name ON celebrities(name);

-- People registered by a patient (dataset/Patient/<patient_id>/<folder>).
-- Searched only by requests carrying that patient_id. Rebuildable index:
-- the API fills and updates it from the folders by itself.
CREATE TABLE IF NOT EXISTS patient_people (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id         TEXT    NOT NULL,                     -- upper-case folder name, e.g. "P001"
    folder              TEXT    NOT NULL,                     -- person folder, e.g. "Son_of_amitab" (shown as "Son of amitab")
    centroid_embedding  BLOB    NOT NULL,                     -- L2-normalized float32 vector (ArcFace, little-endian)
    embedding_dimension INTEGER NOT NULL,
    sample_count        INTEGER NOT NULL,                     -- photos averaged into the centroid
    folder_mtime        REAL    NOT NULL,                     -- folder modification time when embedded; a change triggers re-embedding
    created_at          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patient_id, folder)
);
