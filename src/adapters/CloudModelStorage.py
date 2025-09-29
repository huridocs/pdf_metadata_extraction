from pathlib import Path

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.ports.ModelStorage import ModelStorage


class CloudModelStorage(ModelStorage):

    def __init__(self, google_cloud_storage, logger: Logger):
        self.google_cloud_storage = google_cloud_storage
        self.logger = logger

    def upload_model(self, extraction_identifier: ExtractionIdentifier, source_path: str) -> bool:
        """Upload a trained model to Google Cloud Storage"""
        try:
            if self.google_cloud_storage is None:
                self.logger.log(extraction_identifier, "Google Cloud Storage not available", "error")
                return False

            cloud_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)
            self.google_cloud_storage.copy_to_cloud(Path(source_path), cloud_path)
            self.logger.log(extraction_identifier, f"Model uploaded to cloud storage from {source_path}")
            return True
        except Exception as e:
            self.logger.log(extraction_identifier, f"Model upload failed: {e}", "error")
            return False

    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Download a model from Google Cloud Storage"""
        try:
            if self.google_cloud_storage is None:
                self.logger.log(extraction_identifier, "Google Cloud Storage not available", "error")
                return False

            # Copy model files from cloud to local storage
            cloud_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)
            destination_path = extraction_identifier.get_path()
            self.google_cloud_storage.copy_from_cloud(Path(destination_path), cloud_path)
            self.logger.log(extraction_identifier, f"Model downloaded from cloud storage to {destination_path}")
            return True

        except Exception as e:
            self.logger.log(extraction_identifier, f"Model download failed: {e}", "error")
            return False

    def check_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Check if model upload is completed by checking for completion signal file"""
        try:
            # Check for a completion signal file in cloud storage
            completion_signal_path = Path(extraction_identifier.get_path(), "model_upload_complete.signal")

            if completion_signal_path.exists():
                return True

            # Try to download the signal file from cloud storage
            if self.google_cloud_storage is not None:
                cloud_signal_path = Path(
                    extraction_identifier.run_name, extraction_identifier.extraction_name, "model_upload_complete.signal"
                )
                try:
                    self.google_cloud_storage.copy_from_cloud(completion_signal_path.parent, cloud_signal_path.parent)
                    return completion_signal_path.exists()
                except Exception:
                    # Signal file doesn't exist in cloud storage
                    return False

            return False

        except Exception as e:
            self.logger.log(extraction_identifier, f"Error checking model completion signal: {e}", "error")
            return False

    def create_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Create a completion signal file after successful model upload"""
        try:
            # Create a local signal file
            completion_signal_path = Path(extraction_identifier.get_path(), "model_upload_complete.signal")
            completion_signal_path.parent.mkdir(parents=True, exist_ok=True)

            with open(completion_signal_path, "w") as f:
                f.write("Model upload completed successfully")

            # Upload the signal file to cloud storage
            if self.google_cloud_storage is not None:
                cloud_signal_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)
                try:
                    self.google_cloud_storage.copy_to_cloud(completion_signal_path.parent, cloud_signal_path)
                    self.logger.log(extraction_identifier, "Model completion signal uploaded to cloud storage")
                except Exception as e:
                    self.logger.log(extraction_identifier, f"Failed to upload completion signal to cloud: {e}", "error")

            return True

        except Exception as e:
            self.logger.log(extraction_identifier, f"Error creating model completion signal: {e}", "error")
            return False
