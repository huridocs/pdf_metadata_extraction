from time import time

import redis

from config import config_logger, logs_queue
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LogsMessage import Severity, LogsMessage


def delete_old_messages():
    message = logs_queue.receiveMessage().exceptions(False).execute()

    while message and message["ts"] < time() - 2 * 60 * 60 * 24:
        logs_queue.deleteMessage(id=message["id"]).execute()
        message = logs_queue.receiveMessage().exceptions(False).execute()


def send_logs(extraction_identifier: ExtractionIdentifier, message: str, severity: Severity = Severity.info):
    if severity == Severity.error:
        config_logger.error(message + " for " + extraction_identifier.model_dump_json())
    else:
        config_logger.info(message + " for " + extraction_identifier.model_dump_json())
    try:
        delete_old_messages()
        log_message = LogsMessage(
            tenant=extraction_identifier.run_name,
            extraction_name=extraction_identifier.extraction_name,
            severity=severity,
            message=message,
        )
        logs_queue.sendMessage().message(log_message.dump()).execute()
    except redis.ConnectionError:
        print(severity, message)
    except Exception as e:
        config_logger.error(f"Error sending logs: {e}")
