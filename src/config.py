import os
from os.path import join
from pathlib import Path

NAME = "metadata_extractor"
PARAGRAPH_EXTRACTION_NAME = os.environ.get("PARAGRAPH_EXTRACTION_NAME", f"extract_paragraphs")
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
