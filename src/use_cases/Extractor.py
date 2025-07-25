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
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionData import PredictionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.FilterValidSegmentsPages import FilterValidSegmentsPages
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.send_logs import send_logs

from config import (
    DATA_PATH,
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

if UPLOAD_MODELS_TO_CLOUD_STORAGE:
    server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
    google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)


class Extractor:
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

    @staticmethod
    def get_prediction_samples(
        extractor_identifier: ExtractionIdentifier, prediction_data_list: list[PredictionData] = None
    ) -> list[PredictionSample]:
        filter_valid_pages = FilterValidSegmentsPages(extractor_identifier)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        prediction_samples: list[PredictionSample] = []
        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)
            entity_name = prediction_data.entity_name if prediction_data.entity_name else prediction_data.xml_file_name

            xml_file = XmlFile(
                extraction_identifier=extractor_identifier,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers)
            else:
                pdf_data = PdfData.from_texts([""])

            xml_file.delete()

            sample = PredictionSample(pdf_data=pdf_data, entity_name=entity_name, source_text=prediction_data.source_text)
            prediction_samples.append(sample)

        return prediction_samples

    @staticmethod
    def get_samples_for_training(
        extraction_identifier: ExtractionIdentifier, labeled_data_list: list[LabeledData]
    ) -> list[TrainingSample]:
        multi_option_samples: list[TrainingSample] = list()
        page_numbers_list = FilterValidSegmentsPages(extraction_identifier).for_training(labeled_data_list)
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                extraction_identifier=extraction_identifier,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers_to_keep)
            else:
                pdf_data = PdfData.from_texts([""])
            sample = TrainingSample(
                pdf_data=pdf_data, labeled_data=labeled_data, segment_selector_texts=[labeled_data.source_text]
            )
            multi_option_samples.append(sample)
            xml_file.delete()

        return multi_option_samples

    def create_models(self) -> (bool, str):
        start = time()
        send_logs(self.extraction_identifier, "Loading data to create model")

        if SAMPLES_IN_LOCAL_DB:
            labeled_data_list = self.persistence_repository.load_labeled_data(self.extraction_identifier)
            samples = self.get_samples_for_training(self.extraction_identifier, labeled_data_list)
        else:
            samples = self.import_samples(extraction_identifier=self.extraction_identifier, for_training=True)

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

    def save_suggestions(self, suggestions: list[Suggestion]) -> (bool, str):
        self.persistence_repository.save_suggestions(self.extraction_identifier, suggestions)
        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        if SAMPLES_IN_LOCAL_DB:
            prediction_data_list = self.persistence_repository.load_prediction_data(self.extraction_identifier)
            prediction_samples = self.get_prediction_samples(self.extraction_identifier, prediction_data_list)
        else:
            prediction_samples = self.import_samples(extraction_identifier=self.extraction_identifier, for_training=False)

        if UPLOAD_MODELS_TO_CLOUD_STORAGE and not self.extraction_identifier.get_path().exists():
            try:
                extractor_path = Path(self.extraction_identifier.run_name, self.extraction_identifier.extraction_name)
                google_cloud_storage.copy_from_cloud(extractor_path, Path(DATA_PATH, self.extraction_identifier.run_name))
                config_logger.info(f"Model downloaded from cloud {self.extraction_identifier.get_path()}")
            except:
                config_logger.info(f"No model on cloud {self.extraction_identifier.get_path()}")
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.predict(prediction_samples)

    def save_paragraphs_from_languages(self) -> (bool, str):
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
    def remove_old_models(extractor_identifier: ExtractionIdentifier):
        if exists(extractor_identifier.get_path()):
            os.utime(extractor_identifier.get_path())

        for run_name in os.listdir(DATA_PATH):
            if run_name == "cache":
                continue

            for extraction_name in os.listdir(join(DATA_PATH, run_name)):
                extractor_identifier_to_check = ExtractionIdentifier(
                    run_name=run_name, extraction_name=extraction_name, output_path=DATA_PATH
                )

                if not extractor_identifier_to_check.is_old():
                    continue

                if UPLOAD_MODELS_TO_CLOUD_STORAGE:
                    try:
                        google_cloud_storage.upload_to_cloud(run_name, Path(extractor_identifier_to_check.get_path()))
                        config_logger.info(f"Model uploaded to cloud {extractor_identifier_to_check.get_path()}")
                    except:
                        config_logger.error(f"Error uploading model to cloud {extractor_identifier_to_check.get_path()}")

                config_logger.info(f"Removing old model folder {extractor_identifier_to_check.get_path()}")
                shutil.rmtree(extractor_identifier_to_check.get_path(), ignore_errors=True)

    @staticmethod
    def calculate_task(
        task: TrainableEntityExtractionTask | ParagraphExtractorTask, persistence_repository: PersistenceRepository
    ) -> (bool, str):
        if task.task == Extractor.CREATE_MODEL_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant, extraction_name=task.params.id, metadata=task.params.metadata, output_path=DATA_PATH
            )

            Extractor.remove_old_models(extractor_identifier)

            if task.params.options:
                options = task.params.options
            else:
                options = extractor_identifier.get_options()

            multi_value = task.params.multi_value
            extractor = Extractor(extractor_identifier, persistence_repository, options, multi_value)
            return extractor.create_models()

        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant, extraction_name=task.params.id, metadata=task.params.metadata, output_path=DATA_PATH
            )
            extractor = Extractor(extractor_identifier, persistence_repository)
            suggestions = extractor.get_suggestions()

            if not suggestions:
                return False, "No data to calculate suggestions"

            if SAMPLES_IN_LOCAL_DB:
                return extractor.save_suggestions(suggestions)

            return extractor.send_suggestions(extractor_identifier, suggestions)

        if task.task == PARAGRAPH_EXTRACTION_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=task.key, output_path=DATA_PATH
            )
            extractor = Extractor(extractor_identifier, persistence_repository)
            return extractor.save_paragraphs_from_languages()

        return False, "Error"

    @staticmethod
    def import_samples(
        extraction_identifier: ExtractionIdentifier, for_training: bool
    ) -> list[TrainingSample | PredictionSample]:
        samples: list[TrainingSample | PredictionSample] = list()
        max_retries = 3
        retry_delay = 5  # seconds

        url = f"{SERVICE_HOST}:{SERVICE_PORT}"
        url += "/get_samples_training" if for_training else "/get_samples_prediction"
        url += f"/{extraction_identifier.run_name}/{extraction_identifier.extraction_name}"

        while True:
            try:
                response = requests.get(url)
                response.raise_for_status()

                if not response.json():
                    break

                samples.extend([TrainingSample(**x) if for_training else PredictionSample(**x) for x in response.json()])
            except requests.exceptions.RequestException as e:
                config_logger.error(f"Error fetching training samples: {e}")
                if max_retries > 0:
                    max_retries -= 1
                    config_logger.info(f"Retrying in {retry_delay} seconds...")
                    sleep(retry_delay)
                    continue
                else:
                    config_logger.error("Max retries reached. Exiting.")
                    break

        return samples

    @staticmethod
    def send_suggestions(extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]) -> (bool, str):
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
