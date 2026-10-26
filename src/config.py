import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

# Main dataset folder:
#   dataset/Celebrity/<Name>/*.jpg
#   dataset/Objects/{train,test}/<Object>/*.jpg
#   dataset/Patient/<PatientID>/<Person>/*.jpg
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "Celebrity"
CLEAN_DATASET_DIR = ROOT_DIR / "cleaned_dataset"

# SQLite database holding one centroid embedding per celebrity.
# Created automatically if it does not exist.
DATABASE_PATH = Path(
    os.getenv("CELEBRITY_DB_PATH", ROOT_DIR / "database" / "celebrity_embeddings.db")
)
METADATA_DIR = ROOT_DIR / "metadata"
METADATA_CSV = METADATA_DIR / "celebrities.csv"
QUALITY_REPORT_CSV = METADATA_DIR / "dataset_quality_report.csv"
KAGGLE_MANIFEST_CSV = METADATA_DIR / "kaggle_selection.csv"
METADATA_XLSX = METADATA_DIR / "celebrities.xlsx"
COVERAGE_REPORT_CSV = METADATA_DIR / "coverage_report.csv"

# Minimum clean images a celebrity needs to be recognizable.
MIN_IMAGES_PER_IDENTITY = 10
TEST_CASES_CSV = METADATA_DIR / "test_cases.csv"
TEST_IMAGES_DIR = ROOT_DIR / "test_images"

# People outside the celebrity metadata used as unknown-person tests.
UNKNOWN_PEOPLE_DIR = ROOT_DIR / "unknown_people"
UNKNOWN_TEST_IMAGES = 2

MODEL_NAME = "buffalo_l"
# Small, fast face detector (buffalo_s det_500m, auto-downloaded) at 320.
# ArcFace still comes from MODEL_NAME. For the original detector, set
# DETECTOR_PACK = None and DETECTION_SIZE = (640, 640), then rebuild centroids.
DETECTOR_PACK = "buffalo_s"
DETECTOR_FILE = "det_500m.onnx"
DETECTION_SIZE = (320, 320)
SIMILARITY_THRESHOLD = 0.80

# People registered by a patient. A request with a patient_id is matched
# against that patient's people ONLY (celebrities and objects are never
# searched); a request without one runs the celebrity -> object flow.
# Photos: dataset/Patient/<PatientID>/<Person>/*.jpg
#         (e.g. P001/Son_of_amitab/1.jpg, shown as "Son of amitab")
PATIENT_DATASET_DIR = DATASET_DIR / "Patient"
# Lower than the celebrity threshold: a patient registers a few photos per
# person, and a centroid of few photos scores lower for the same person.
# Measured on 20 people (held-out photos): same person, 3-10 photos ->
# median 0.71-0.77, 5th percentile 0.49-0.58; different people never above
# 0.28. At 0.80 only 13-28% of genuine photos would be accepted.
PATIENT_SIMILARITY_THRESHOLD = 0.80
MIN_IMAGES_PER_PATIENT_PERSON = 1

# General object recognition (CLIP). Kept in its own table: CLIP and ArcFace
# embeddings live in different feature spaces and are never compared.
OBJECT_DATASET_DIR = DATASET_DIR / "Objects"
OBJECT_TRAIN_DIR = OBJECT_DATASET_DIR / "train"
OBJECT_TEST_DIR = OBJECT_DATASET_DIR / "test"
OBJECT_METADATA_XLSX = METADATA_DIR / "objects.xlsx"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP image model exported to ONNX on first use (~350 MB, not in git):
# same embeddings as PyTorch, about 2x faster on CPU.
CLIP_ONNX_PATH = ROOT_DIR / "models" / "clip_vision.onnx"
OBJECT_SIMILARITY_THRESHOLD = 0.80

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MIN_IMAGE_SIDE = 160
MIN_FACE_AREA_RATIO = 0.03
MIN_BLUR_SCORE = 50.0

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
