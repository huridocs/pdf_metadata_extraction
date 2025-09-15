import os
import shutil
import pickle
from os.path import join, exists
from pathlib import Path
from time import time

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.send_logs import send_logs

from config import (
    MODELS_DATA_PATH,
    LAST_RUN_PATH,
)
from domain.TasksNames import TasksNames
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from ports.PersistenceRepository import PersistenceRepository
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase

google_cloud_storage = None
try:
    server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
    google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
    config_logger.info("Google Cloud Storage client initialized successfully")
except Exception as e:
    config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
    google_cloud_storage = None


class ExtractorUseCase:
    def __init__(
        self,
        extraction_identifier: ExtractionIdentifier,
        persistence_repository: PersistenceRepository,
        options: list[Option] = None,
        multi_value: bool = False,
    ):
        self.extraction_identifier = extraction_identifier
        self.persistence_repository = persistence_repository
        self.multi_value = multi_value
        self.options = options
        self.sample_processor = SampleProcessorUseCase(extraction_identifier, persistence_repository)

    def create_models(self) -> tuple[bool, str]:
        start = time()
        send_logs(self.extraction_identifier, "Loading data to create model")

        samples = self.sample_processor.get_training_samples()

        extraction_data: ExtractionData = ExtractionData(
            samples=samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

        self.save_last_extraction_data(extraction_data)

        self.delete_training_data()
        send_logs(self.extraction_identifier, f"Set data in {round(time() - start, 2)} seconds")
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.train(extraction_data)

    def save_last_extraction_data(self, extraction_data: ExtractionData):
        try:
            if LAST_RUN_PATH.exists():
                shutil.rmtree(LAST_RUN_PATH, ignore_errors=True)

            LAST_RUN_PATH.mkdir(parents=True, exist_ok=True)
            extraction_name = f"{self.extraction_identifier.run_name}_{self.extraction_identifier.extraction_name}.pickle"
            path = LAST_RUN_PATH / extraction_name
            with open(path, "wb") as file:
                pickle.dump(extraction_data, file)
            send_logs(self.extraction_identifier, f"Saved last run at {path}")
        except Exception as e:
            send_logs(self.extraction_identifier, f"Error saving extraction_data for debugging: {e}")

    def delete_training_data(self):
        training_xml_path = XmlFile(extraction_identifier=self.extraction_identifier, to_train=True).xml_folder_path
        send_logs(self.extraction_identifier, f"Deleting training data in {training_xml_path}")
        shutil.rmtree(training_xml_path, ignore_errors=True)

    def save_suggestions(self, suggestions: list[Suggestion]) -> tuple[bool, str]:
        self.persistence_repository.save_suggestions(self.extraction_identifier, suggestions)
        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        prediction_samples = self.sample_processor.get_prediction_samples_for_suggestions()

        if (google_cloud_storage is not None
            and not exists(self.extraction_identifier.get_path())):
            try:
                extractor_path = Path(self.extraction_identifier.run_name, self.extraction_identifier.extraction_name)
                google_cloud_storage.copy_from_cloud(
                    extractor_path, Path(MODELS_DATA_PATH, self.extraction_identifier.run_name)
                )
                config_logger.info(f"Model downloaded from cloud {self.extraction_identifier.get_path()}")
            except Exception as e:
                config_logger.error(f"Error downloading model from cloud: {e}")
                config_logger.warning(f"No model available on cloud for {self.extraction_identifier.get_path()}")

        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.predict(prediction_samples)

    @staticmethod
    def remove_old_models(extractor_identifier_to_keep: ExtractionIdentifier):
        if exists(extractor_identifier_to_keep.get_path()):
            os.utime(extractor_identifier_to_keep.get_path())

        for run_name in [x for x in os.listdir(MODELS_DATA_PATH) if x != "cache"]:
            for extraction_name in os.listdir(join(MODELS_DATA_PATH, run_name)):
                if not Path(MODELS_DATA_PATH, run_name, extraction_name).is_dir():
                    continue

                extractor_identifier_to_check = ExtractionIdentifier(
                    run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
                )

                if extractor_identifier_to_check.is_training_canceled():
                    ExtractorUseCase._handle_canceled_model(extractor_identifier_to_check, run_name, extraction_name)
                    continue

                if not extractor_identifier_to_check.is_old():
                    continue

                ExtractorUseCase._handle_local_model_deletion(extractor_identifier_to_check, True)
                return

    @staticmethod
    def _handle_canceled_model(extractor_identifier: ExtractionIdentifier, run_name: str, extraction_name: str):
        try:
            google_cloud_storage.delete_from_cloud(run_name, extraction_name)
            config_logger.info(f"Delete model from cloud {extractor_identifier.get_path()}")
        except Exception as e:
            config_logger.error(f"Error deleting model from cloud {extractor_identifier.get_path()}: {e}")

    @staticmethod
    def _handle_local_model_deletion(extractor_identifier: ExtractionIdentifier, should_delete_local: bool):
        """Handle local model deletion based on cloud upload success"""
        if should_delete_local:
            config_logger.info(f"Removing old model folder {extractor_identifier.get_path()}")
            shutil.rmtree(extractor_identifier.get_path(), ignore_errors=True)
        else:
            config_logger.info(f"Keeping model locally due to cloud upload failure: {extractor_identifier.get_path()}")

    @staticmethod
    def execute_task(
        task: TrainableEntityExtractionTask, persistence_repository: PersistenceRepository
    ) -> tuple[bool, str]:
        if task.task == TasksNames.CREATE_MODEL_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant,
                extraction_name=task.params.id,
                metadata=task.params.metadata,
                output_path=MODELS_DATA_PATH,
            )

            ExtractorUseCase.remove_old_models(extractor_identifier)

            if task.params.options:
                options = task.params.options
            else:
                options = extractor_identifier.get_options()

            multi_value = task.params.multi_value
            extractor = ExtractorUseCase(extractor_identifier, persistence_repository, options, multi_value)
            return extractor.create_models()

        if task.task == TasksNames.SUGGESTIONS_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant,
                extraction_name=task.params.id,
                metadata=task.params.metadata,
                output_path=MODELS_DATA_PATH,
            )
            extractor = ExtractorUseCase(extractor_identifier, persistence_repository)
            suggestions = extractor.get_suggestions()

            if not suggestions:
                return False, "No data to calculate suggestions"

            return extractor.save_suggestions(suggestions)

        return False, "Error"
