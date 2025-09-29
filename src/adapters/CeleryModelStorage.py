from pathlib import Path
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs

from trainable_entity_extractor_abstractions.interfaces.ModelStorage import ModelStorage
from drivers.distributed_worker.distributed_no_gpu import upload_model


class CeleryModelStorage(ModelStorage):
    """Celery/Google Cloud Storage implementation of the ModelStorage interface"""

    def __init__(self, google_cloud_storage):
        self.google_cloud_storage = google_cloud_storage

    def upload_model(
        self, extraction_identifier: ExtractionIdentifier, method_name: str, extractor_job: TrainableEntityExtractorJob
    ) -> bool:
        """Upload a trained model using Celery broadcast to all workers"""
        try:
            # Use broadcast to send upload task to all Celery workers
            upload_model.broadcast(extraction_identifier, method_name, extractor_job)
            send_logs(extraction_identifier, f"Upload task broadcasted to all workers for method {method_name}")
            return True
        except Exception as e:
            send_logs(extraction_identifier, f"Upload broadcast failed: {e}", LogSeverity.error)
            return False

    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Download a model from Google Cloud Storage"""
        try:
            if self.google_cloud_storage is None:
                send_logs(extraction_identifier, "Google Cloud Storage not available", LogSeverity.error)
                return False

            # Copy model files from cloud to local storage
            local_path = Path(extraction_identifier.get_path())
            cloud_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)

            self.google_cloud_storage.copy_from_cloud(local_path.parent, cloud_path)
            send_logs(extraction_identifier, f"Model downloaded from cloud storage")
            return True

        except Exception as e:
            send_logs(extraction_identifier, f"Model download failed: {e}", LogSeverity.error)
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
            send_logs(extraction_identifier, f"Error checking model completion signal: {e}", LogSeverity.error)
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
                    send_logs(extraction_identifier, "Model completion signal uploaded to cloud storage")
                except Exception as e:
                    send_logs(extraction_identifier, f"Failed to upload completion signal to cloud: {e}", LogSeverity.error)

            return True

        except Exception as e:
            send_logs(extraction_identifier, f"Error creating model completion signal: {e}", LogSeverity.error)
            return False
