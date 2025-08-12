import os
import shutil
import pickle
from os.path import join, exists
from pathlib import Path
from time import time, sleep

import requests
from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.send_logs import send_logs

from config import (
    MODELS_DATA_PATH,
    PARAGRAPH_EXTRACTION_NAME,
    SERVICE_HOST,
    SERVICE_PORT,
    SAMPLES_IN_LOCAL_DB,
    UPLOAD_MODELS_TO_CLOUD_STORAGE,
    LAST_RUN_PATH,
)
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from ports.PersistenceRepository import PersistenceRepository
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase

google_cloud_storage = None
if UPLOAD_MODELS_TO_CLOUD_STORAGE:
    try:
        server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
        google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
        config_logger.info("Google Cloud Storage client initialized successfully")
    except Exception as e:
        config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
        google_cloud_storage = None


class ExtractorUseCase:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

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

        if (
            UPLOAD_MODELS_TO_CLOUD_STORAGE
            and google_cloud_storage is not None
            and not exists(self.extraction_identifier.get_path())
        ):
            try:
                extractor_path = Path(self.extraction_identifier.run_name, self.extraction_identifier.extraction_name)
                google_cloud_storage.copy_from_cloud(
                    extractor_path, Path(MODELS_DATA_PATH, self.extraction_identifier.run_name)
                )
                config_logger.info(f"Model downloaded from cloud {self.extraction_identifier.get_path()}")
            except Exception as e:
                config_logger.error(f"Error downloading model from cloud: {e}")
                config_logger.warning(f"No model available on cloud for {self.extraction_identifier.get_path()}")
                # Continue without the model - let TrainableEntityExtractor handle the missing model

        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.predict(prediction_samples)

    def save_paragraphs_from_languages(self) -> tuple[bool, str]:
        paragraph_extraction_data = self.persistence_repository.load_paragraph_extraction_data(self.extraction_identifier)
        if not paragraph_extraction_data:
            return False, "No data to extract paragraphs"

        paragraphs_from_languages = self.get_paragraphs_from_languages(paragraph_extraction_data)

        aligner_use_case = MultilingualParagraphAlignerUseCase(self.extraction_identifier)
        aligner_use_case.align_languages(paragraphs_from_languages)

        for paragraphs_from_language in paragraphs_from_languages:
            self.persistence_repository.save_paragraphs_from_language(
                self.extraction_identifier, paragraphs_from_language.to_db()
            )

        return True, ""

    def get_paragraphs_from_languages(self, paragraph_extraction_data):
        paragraphs_from_languages: list[ParagraphsFromLanguage] = list()
        for xml_segments in paragraph_extraction_data.xmls:
            segmentation_data = SegmentationData(
                page_width=0, page_height=0, xml_segments_boxes=xml_segments.xml_segments_boxes, label_segments_boxes=[]
            )
            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier, to_train=True, xml_file_name=xml_segments.xml_file_name
            )
            pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
            paragraphs_from_language = ParagraphsFromLanguage(
                language=xml_segments.language,
                paragraphs=[ParagraphFeatures.from_pdf_data(pdf_data, x) for x in pdf_data.pdf_data_segments],
                is_main_language=xml_segments.is_main_language,
            )
            paragraphs_from_languages.append(paragraphs_from_language)
            xml_file.delete()

        return paragraphs_from_languages

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

                should_delete_local = ExtractorUseCase._upload_model_to_cloud(extractor_identifier_to_check, run_name)
                ExtractorUseCase._handle_local_model_deletion(extractor_identifier_to_check, should_delete_local)
                return

    @staticmethod
    def _handle_canceled_model(extractor_identifier: ExtractionIdentifier, run_name: str, extraction_name: str):
        if UPLOAD_MODELS_TO_CLOUD_STORAGE and google_cloud_storage is not None:
            try:
                google_cloud_storage.delete_from_cloud(run_name, extraction_name)
                config_logger.info(f"Delete model from cloud {extractor_identifier.get_path()}")
            except Exception as e:
                config_logger.error(f"Error deleting model from cloud {extractor_identifier.get_path()}: {e}")

    @staticmethod
    def _upload_model_to_cloud(extractor_identifier: ExtractionIdentifier, run_name: str) -> bool:
        should_delete_local = True

        if UPLOAD_MODELS_TO_CLOUD_STORAGE and google_cloud_storage is not None:
            try:
                google_cloud_storage.upload_to_cloud(run_name, Path(extractor_identifier.get_path()))
                config_logger.info(f"Model uploaded to cloud {extractor_identifier.get_path()}")
            except Exception as e:
                config_logger.error(f"Error uploading model to cloud {extractor_identifier.get_path()}: {e}")
                should_delete_local = False  # Don't delete if upload failed
                config_logger.warning(f"Keeping local model due to failed cloud upload: {extractor_identifier.get_path()}")

        return should_delete_local

    @staticmethod
    def _handle_local_model_deletion(extractor_identifier: ExtractionIdentifier, should_delete_local: bool):
        """Handle local model deletion based on cloud upload success"""
        if should_delete_local:
            config_logger.info(f"Removing old model folder {extractor_identifier.get_path()}")
            shutil.rmtree(extractor_identifier.get_path(), ignore_errors=True)
        else:
            config_logger.info(f"Keeping model locally due to cloud upload failure: {extractor_identifier.get_path()}")

    @staticmethod
    def calculate_task(
        task: TrainableEntityExtractionTask | ParagraphExtractorTask, persistence_repository: PersistenceRepository
    ) -> tuple[bool, str]:
        if task.task == ExtractorUseCase.CREATE_MODEL_TASK_NAME:
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

        if task.task == ExtractorUseCase.SUGGESTIONS_TASK_NAME:
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

            if SAMPLES_IN_LOCAL_DB:
                return extractor.save_suggestions(suggestions)

            return extractor.send_suggestions(extractor_identifier, suggestions)

        if task.task == PARAGRAPH_EXTRACTION_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=task.key, output_path=MODELS_DATA_PATH
            )
            extractor = ExtractorUseCase(extractor_identifier, persistence_repository)
            return extractor.save_paragraphs_from_languages()

        return False, "Error"

    @staticmethod
    def send_suggestions(extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]) -> tuple[bool, str]:
        max_retries = 3
        retry_delay = 5  # seconds

        url = f"{SERVICE_HOST}:{SERVICE_PORT}"
        url += "/save_suggestions"
        url += f"/{extraction_identifier.run_name}/{extraction_identifier.extraction_name}"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        json_data = [x.model_dump() for x in suggestions]

        while True:
            try:
                response = requests.post(url, headers=headers, json=json_data)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                config_logger.error(f"Error fetching training samples: {e}")
                if max_retries > 0:
                    max_retries -= 1
                    config_logger.info(f"Retrying in {retry_delay} seconds...")
                    sleep(retry_delay)
                    continue
                else:
                    config_logger.error("Max retries reached. Exiting.")
                    return False, "Could not send suggestions back"

        return True, ""
