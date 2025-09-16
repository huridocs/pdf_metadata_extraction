import os
from queue_processor.QueueProcessor import QueueProcessor
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk
from trainable_entity_extractor.config import config_logger
from config import (
    REDIS_HOST,
    REDIS_PORT,
    QUEUES_NAMES,
)
from drivers.queues_processor.MetadataExtractorQueueProcessor import MetadataExtractorQueueProcessor


if __name__ == "__main__":
    try:
        sentry_sdk.init(
            os.environ.get("SENTRY_DSN"),
            traces_sample_rate=0.1,
            environment=os.environ.get("ENVIRONMENT", "development"),
            integrations=[RedisIntegration()],
        )
    except Exception:
        pass

    queues_names = QUEUES_NAMES.split(" ")
    queue_processor = QueueProcessor(REDIS_HOST, REDIS_PORT, queues_names, config_logger)

    process = MetadataExtractorQueueProcessor()
    queue_processor.start(process)
