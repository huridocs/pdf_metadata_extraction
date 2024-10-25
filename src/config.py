import logging
import os
from os.path import join
from pathlib import Path

import graypy


QUEUES_NAMES = os.environ.get("QUEUES_NAMES", "information_extraction")

SERVICE_HOST = os.environ.get("SERVICE_HOST", "http://127.0.0.1")
SERVICE_PORT = os.environ.get("SERVICE_PORT", "5056")
GRAYLOG_IP = os.environ.get("GRAYLOG_IP")
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
MONGO_HOST = os.environ.get("MONGO_HOST", "mongodb://127.0.0.1")
MONGO_PORT = os.environ.get("MONGO_PORT", "29017")
SENTRY_DSN = os.environ.get("SENTRY_DSN")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

APP_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = join(ROOT_PATH, "models_data")

HUGGINGFACE_PATH = join(ROOT_PATH, "huggingface")

handlers = [logging.StreamHandler()]

if GRAYLOG_IP:
    handlers.append(graypy.GELFUDPHandler(GRAYLOG_IP, 12201, localname="pdf_metadata_extraction"))

logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
config_logger = logging.getLogger(__name__)
