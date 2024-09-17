from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LogsMessage import Severity


def send_logs(extraction_identifier: ExtractionIdentifier, message: str, severity: Severity = Severity.info):
    if severity == Severity.error:
        config_logger.error(message + " for " + extraction_identifier.model_dump_json())
    else:
        config_logger.info(message + " for " + extraction_identifier.model_dump_json())
