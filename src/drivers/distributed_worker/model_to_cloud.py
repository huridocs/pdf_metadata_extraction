from pathlib import Path

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

def download_model_from_cloud(self) -> bool:
    try:
        extractor_path = Path(self.extraction_identifier.run_name, self.extraction_identifier.extraction_name)
        google_cloud_storage.copy_from_cloud(
            extractor_path, Path(MODELS_DATA_PATH, self.extraction_identifier.run_name)
        )
        config_logger.info(f"Model downloaded from cloud {self.extraction_identifier.get_path()}")
        return True
    except Exception as e:
        config_logger.error(f"Error downloading model from cloud: {e}")
        config_logger.warning(f"No model available on cloud for {self.extraction_identifier.get_path()}")
        return False