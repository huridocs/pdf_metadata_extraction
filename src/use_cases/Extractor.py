import os
import shutil
from os.path import join, exists
from pathlib import Path
from time import time

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.FilterValidSegmentsPages import FilterValidSegmentsPages
from trainable_entity_extractor.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.SegmentationData import SegmentationData
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.send_logs import send_logs

from config import DATA_PATH, PARAGRAPH_EXTRACTION_NAME
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from ports.PersistenceRepository import PersistenceRepository


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

    def get_extraction_data_for_training(self, labeled_data_list: list[LabeledData]) -> ExtractionData:
        multi_option_samples: list[TrainingSample] = list()
        page_numbers_list = FilterValidSegmentsPages(self.extraction_identifier).for_training(labeled_data_list)
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier,
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

        return ExtractionData(
            samples=multi_option_samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

    def create_models(self) -> (bool, str):
        start = time()
        send_logs(self.extraction_identifier, "Loading data to create model")
        labeled_data_list = self.persistence_repository.load_labeled_data(self.extraction_identifier)
        extraction_data: ExtractionData = self.get_extraction_data_for_training(labeled_data_list)
        send_logs(self.extraction_identifier, f"Set data in {round(time() - start, 2)} seconds")
        self.delete_training_data()
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.train(extraction_data)

    def get_prediction_samples(self, prediction_data_list: list[PredictionData] = None) -> list[PredictionSample]:
        filter_valid_pages = FilterValidSegmentsPages(self.extraction_identifier)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        prediction_samples: list[PredictionSample] = []
        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)
            entity_name = prediction_data.entity_name if prediction_data.entity_name else prediction_data.xml_file_name

            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers)
            else:
                pdf_data = PdfData.from_texts([""])

            sample = PredictionSample(pdf_data=pdf_data, entity_name=entity_name, source_text=prediction_data.source_text)
            prediction_samples.append(sample)

        return prediction_samples

    def delete_training_data(self):
        training_xml_path = XmlFile(extraction_identifier=self.extraction_identifier, to_train=True).xml_folder_path
        send_logs(self.extraction_identifier, f"Deleting training data in {training_xml_path}")
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.persistence_repository.delete_labeled_data(self.extraction_identifier)

    def save_suggestions(self, suggestions: list[Suggestion]) -> (bool, str):
        if not suggestions:
            return False, "No data to calculate suggestions"

        self.persistence_repository.save_suggestions(self.extraction_identifier, suggestions)
        xml_folder_path = XmlFile(extraction_identifier=self.extraction_identifier, to_train=False).xml_folder_path
        for suggestion in suggestions:
            entity_name = {"entity_name": suggestion.entity_name, "xml_file_name": ""}
            xml_file_name = {"xml_file_name": suggestion.xml_file_name, "entity_name": ""}
            self.persistence_repository.delete_prediction_data(self.extraction_identifier, [entity_name, xml_file_name])
            path = Path(join(xml_folder_path, suggestion.xml_file_name))
            if not path.is_dir():
                path.unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        prediction_data_list = self.persistence_repository.load_prediction_data(self.extraction_identifier)
        prediction_samples = self.get_prediction_samples(prediction_data_list)
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
            self.persistence_repository.save_paragraphs_from_language(self.extraction_identifier, paragraphs_from_language)

        return True, ""

    def get_paragraphs_from_languages(self, paragraph_extraction_data):
        paragraphs_from_languages: list[ParagraphsFromLanguage] = list()
        for xml_segments in paragraph_extraction_data.xmls_segments:
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
                if extractor_identifier_to_check.is_old():
                    config_logger.info(f"Removing old model folder {extractor_identifier_to_check.get_path()}")
                    shutil.rmtree(extractor_identifier_to_check.get_path(), ignore_errors=True)

    @staticmethod
    def calculate_task(
        task: TrainableEntityExtractionTask | ParagraphExtractorTask, persistence_repository: PersistenceRepository
    ) -> (bool, str):
        if task.task == Extractor.CREATE_MODEL_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant, extraction_name=task.extraction_name, metadata=task.metadata, output_path=DATA_PATH
            )

            Extractor.remove_old_models(extractor_identifier)

            if task.xmls.options:
                options = task.xmls.options
            else:
                options = extractor_identifier.get_options()

            multi_value = task.xmls.multi_value
            extractor = Extractor(extractor_identifier, persistence_repository, options, multi_value)
            return extractor.create_models()

        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant, extraction_name=task.extraction_name, metadata=task.metadata, output_path=DATA_PATH
            )
            extractor = Extractor(extractor_identifier, persistence_repository)
            suggestions = extractor.get_suggestions()
            return extractor.save_suggestions(suggestions)

        if task.task == PARAGRAPH_EXTRACTION_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=task.key, output_path=DATA_PATH
            )
            extractor = Extractor(extractor_identifier, persistence_repository)
            return extractor.save_paragraphs_from_languages()

        return False, "Error"
