import os
from os.path import join
from pathlib import Path

NAME="metadata_extractor"
UPLOAD_MODELS_TO_CLOUD_STORAGE = os.environ.get("UPLOAD_MODELS_TO_CLOUD_STORAGE", "false").lower().strip() == "true"
CALCULATE_MODELS_LOCALLY = os.environ.get("CALCULATE_MODELS_LOCALLY", "true").lower().strip() == "true"
RESTART_IF_NO_GPU = os.environ.get("RESTART_IF_NO_GPU", "false").lower().strip() == "true"
EXECUTE_PARAGRAPH_EXTRACTION = os.environ.get("EXECUTE_PARAGRAPH_EXTRACTION", "true").lower().strip() == "true"

if EXECUTE_PARAGRAPH_EXTRACTION:
    PARAGRAPH_EXTRACTION_NAME = os.environ.get("PARAGRAPH_EXTRACTION_NAME", f"extract_paragraphs")
else:
    PARAGRAPH_EXTRACTION_NAME = f"not_used_extract_paragraphs"

QUEUES_NAMES = os.environ.get("QUEUES_NAMES", f"information_extraction")
QUEUES_NAMES += f" {PARAGRAPH_EXTRACTION_NAME}"

SERVICE_HOST = os.environ.get("SERVICE_HOST", "http://127.0.0.1")
SERVICE_PORT = os.environ.get("SERVICE_PORT", "5056")
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
MONGO_HOST = os.environ.get("MONGO_HOST", "mongodb://127.0.0.1")
MONGO_PORT = os.environ.get("MONGO_PORT", "29017")
SENTRY_DSN = os.environ.get("SENTRY_DSN")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

APP_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()
MODELS_DATA_PATH = join(ROOT_PATH, "models_data")
DATA_PATH = join(ROOT_PATH, "data")
LAST_RUN_PATH = Path(DATA_PATH, "last_run_data")
