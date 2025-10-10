import unittest
from unittest.mock import MagicMock
import shutil
from pathlib import Path

from adapters.CloudModelStorage import CloudModelStorage
from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from config import MODELS_DATA_PATH


class TestCloudModelStorage(unittest.TestCase):

    def setUp(self):
        server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
        if GoogleCloudStorage.could_be_configured():
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
        else:
            self.google_cloud_storage = None
        self.logger = MagicMock()
        self.cloud_storage = CloudModelStorage(self.google_cloud_storage, self.logger)

    @unittest.skipIf(GoogleCloudStorage.could_be_configured() is False, "Google Cloud Storage not configured")
    def test_upload_model(self):
        extraction_identifier = ExtractionIdentifier(
            run_name="test_run", extraction_name="test_extraction", output_path=MODELS_DATA_PATH
        )

        base_path = Path(extraction_identifier.get_path())
        base_path.mkdir(parents=True, exist_ok=True)

        (base_path / "model_file_1.bin").write_text("model data 1")
        (base_path / "model_file_2.pt").write_text("model data 2")
        (base_path / "config.json").write_text('{"config": "value"}')

        models_dir = base_path / "models"
        models_dir.mkdir(exist_ok=True)
        (models_dir / "checkpoint_1.bin").write_text("checkpoint 1 data")
        (models_dir / "checkpoint_2.bin").write_text("checkpoint 2 data")

        nested_dir = models_dir / "nested" / "deep" / "structure"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "deep_file.txt").write_text("deep nested file")
        (nested_dir / "another_deep_file.json").write_text('{"deep": "json"}')

        tokenizer_dir = base_path / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "vocab.txt").write_text("vocab data")
        (tokenizer_dir / "tokenizer_config.json").write_text('{"tokenizer": "config"}')

        metadata_dir = base_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / "training_stats.csv").write_text("epoch,loss\n1,0.5\n2,0.3")

        extractor_job = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="test_extractor",
            method_name="test_method",
            gpu_needed=False,
            timeout=3600,
        )
        result = self.cloud_storage.upload_model(extraction_identifier, extractor_job)
        self.assertTrue(result)
        self.logger.log.assert_called()

    @unittest.skipIf(GoogleCloudStorage.could_be_configured() is False, "Google Cloud Storage not configured")
    def test_download_model(self):
        extraction_identifier = ExtractionIdentifier(
            run_name="test_run", extraction_name="test_extraction", output_path=MODELS_DATA_PATH
        )

        if Path(extraction_identifier.get_path()).exists():
            shutil.rmtree(extraction_identifier.get_path())

        download_result = self.cloud_storage.download_model(extraction_identifier)
        self.assertTrue(download_result)

        self.assertTrue(Path(extraction_identifier.get_path()).exists())

        downloaded_job = self.cloud_storage.get_extractor_job(extraction_identifier)
        self.assertIsNotNone(downloaded_job)

    def test_get_extractor_job_from_cloud(self):
        extraction_identifier = ExtractionIdentifier(
            run_name="right", extraction_name="68e9016daab14476dcab7ab0", output_path=MODELS_DATA_PATH
        )

        local_path = Path(extraction_identifier.get_path())
        if local_path.exists():
            shutil.rmtree(local_path)

        extractor_job = self.cloud_storage.get_extractor_job(extraction_identifier)

        self.assertIsNotNone(extractor_job, "Extractor job should be downloaded from cloud")
        self.assertEqual(extractor_job.run_name, "right")
        self.assertEqual(extractor_job.extraction_name, "68e9016daab14476dcab7ab0")
        self.assertIsNotNone(extractor_job.extractor_name)
        self.assertIsNotNone(extractor_job.method_name)

        self.assertTrue(local_path.exists(), "Extractor job file should exist locally after download")
