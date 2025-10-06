import json
from pathlib import Path
from typing import Optional

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.ports.ModelStorage import ModelStorage
from trainable_entity_extractor.config import EXTRACTOR_JOB_PATH
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage


class CloudModelStorage(ModelStorage):

    def __init__(self, google_cloud_storage: GoogleCloudStorage | None, logger: Logger):
        self.google_cloud_storage = google_cloud_storage
        self.logger = logger

    def upload_model(self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob) -> bool:
        try:
            if not self.save_extractor_job(extraction_identifier, extractor_job):
                self.logger.log(
                    extraction_identifier, "Failed to save extractor job locally before upload", LogSeverity.error
                )
                return False

            if self.google_cloud_storage is None:
                self.logger.log(extraction_identifier, "Google Cloud Storage not available")
                return True

            source_path = Path(extraction_identifier.get_path())
            cloud_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)
            self.google_cloud_storage.upload_to_cloud(str(cloud_path), source_path)
            self.logger.log(extraction_identifier, f"Model uploaded to cloud storage from {source_path}")
            return True
        except Exception as e:
            self.logger.log(extraction_identifier, f"Model upload failed: {e}", LogSeverity.error)
            return False

    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        destination_path = extraction_identifier.get_path()

        if self.model_exists_locally(extraction_identifier):
            return True

        try:
            if self.google_cloud_storage is None:
                self.logger.log(extraction_identifier, "Google Cloud Storage not available")
                return False

            cloud_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name)

            if destination_path is None:
                destination_path = extraction_identifier.get_path()

            self.google_cloud_storage.copy_from_cloud(Path(destination_path), cloud_path)
            self.logger.log(extraction_identifier, f"Model downloaded from cloud storage to {destination_path}")
            return True

        except Exception as e:
            self.logger.log(extraction_identifier, f"Model download failed: {e}", LogSeverity.error)
            return False

    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> Optional[TrainableEntityExtractorJob]:
        try:
            local_path = Path(extraction_identifier.get_path(), EXTRACTOR_JOB_PATH)
            if local_path.exists():
                return LocalModelStorage().get_extractor_job(extraction_identifier)

            if self.google_cloud_storage is None:
                self.logger.log(extraction_identifier, "Google Cloud Storage not available", "error")
                return None

            cloud_job_path = Path(extraction_identifier.run_name, extraction_identifier.extraction_name, EXTRACTOR_JOB_PATH)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.google_cloud_storage.copy_from_cloud(cloud_job_path, local_path)

            with open(local_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
                return self.deserialize_job_from_dict(job_data)

        except Exception as e:
            self.logger.log(extraction_identifier, f"Failed to get extractor job from cloud: {e}", "error")
            return None

    @staticmethod
    def model_exists_locally(extraction_identifier: ExtractionIdentifier) -> bool:
        model_path = Path(extraction_identifier.get_path())
        if not model_path.exists() or not model_path.is_dir():
            return False

        files = list(model_path.iterdir())
        if not files:
            return False

        for file in files:
            if file.name != EXTRACTOR_JOB_PATH.parent:
                return True

        return False
