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
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:25018/pdf_metadata_extraction")
SENTRY_DSN = os.environ.get("SENTRY_DSN")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
MATERIALS_HOURS_TO_KEEP = int(os.environ.get("MATERIALS_HOURS_TO_KEEP", "6"))

APP_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()
MODELS_DATA_PATH = join(ROOT_PATH, "models_data")
DATA_PATH = join(ROOT_PATH, "data")
LAST_RUN_PATH = Path(DATA_PATH, "last_run_data")
NO_GPU = os.environ.get("NO_GPU", "false").lower().strip() == "true"

# Flair based NER text-to-text methods are currently disabled: the installed Flair/torch
# combination cannot load the `ner-ontonotes-large` checkpoint (torch>=2.6 defaults
# `weights_only=True`), which makes the methods fail and stalls the first training run on a
# ~2.2 GB model download. The method code is kept in trainable_entity_extractor; set this to
# `true` to register them again once the dependency issue is resolved.
FLAIR_NER_METHODS_ENABLED = os.environ.get("FLAIR_NER_METHODS_ENABLED", "false").lower().strip() == "true"
