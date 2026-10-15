import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATASET_DIR = ROOT_DIR / "celebrity_dataset"
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

# General object recognition (CLIP). Kept in its own table: CLIP and ArcFace
# embeddings live in different feature spaces and are never compared.
OBJECT_TRAIN_DIR = ROOT_DIR / "object_dataset" / "train"
OBJECT_TEST_DIR = ROOT_DIR / "object_dataset" / "test"
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
