from pathlib import Path
import tempfile

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier

from config import MODELS_DATA_PATH

try:
    server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
    google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
    config_logger.info("Google Cloud Storage client initialized successfully")
except Exception as e:
    config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
    google_cloud_storage = None


def upload_model_to_cloud(extractor_identifier: ExtractionIdentifier, run_name: str) -> bool:
    try:
        google_cloud_storage.upload_to_cloud(run_name, Path(extractor_identifier.get_path()))
        config_logger.info(f"Model uploaded to cloud {extractor_identifier.get_path()}")
    except Exception as e:
        config_logger.error(f"Error uploading model to cloud {extractor_identifier.get_path()}: {e}")
        config_logger.warning(f"Keeping local model due to failed cloud upload: {extractor_identifier.get_path()}")
        return False

    return True


def upload_completion_signal(extractor_identifier: ExtractionIdentifier, run_name: str) -> bool:
    """Upload a completion signal file to indicate model upload is complete"""
    try:
        # Create a temporary completion signal file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.completed', delete=False) as temp_file:
            temp_file.write(f"Model upload completed for {extractor_identifier.get_path()}")
            temp_file_path = temp_file.name

        # Upload the completion signal file
        completion_file_path = Path(temp_file_path)
        google_cloud_storage.upload_file_to_cloud(run_name, completion_file_path, f"{extractor_identifier.extraction_name}.completed")

        # Clean up temp file
        Path(temp_file_path).unlink()

        config_logger.info(f"Completion signal uploaded for {extractor_identifier.get_path()}")
        return True
    except Exception as e:
        config_logger.error(f"Error uploading completion signal: {e}")
        return False


def check_model_completion_signal(extractor_identifier: ExtractionIdentifier) -> bool:
    """Check if model upload completion signal exists in cloud"""
    try:
        completion_file_name = f"{extractor_identifier.extraction_name}.completed"
        # This assumes GoogleCloudStorage has a method to check file existence
        # If not available, we'll need to implement it differently
        return google_cloud_storage.file_exists(extractor_identifier.run_name, completion_file_name)
    except Exception as e:
        config_logger.error(f"Error checking completion signal: {e}")
        return False


def download_model_from_cloud(extraction_identifier: ExtractionIdentifier) -> bool:
    try:
        extractor_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)
        google_cloud_storage.copy_from_cloud(extractor_path, Path(MODELS_DATA_PATH, extraction_identifier.run_name))
        config_logger.info(f"Model downloaded from cloud {extraction_identifier.get_path()}")
        return True
    except Exception as e:
        config_logger.error(f"Error downloading model from cloud: {e}")
        config_logger.warning(f"No model available on cloud for {extraction_identifier.get_path()}")
        return False
