import os
import re
import shutil
from os.path import join, exists
from pathlib import Path
from time import time

import pymongo
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

from config import MONGO_PORT, MONGO_HOST, DATA_PATH, PARAGRAPH_EXTRACTION_NAME
from data.ParagraphExtractionData import ParagraphExtractionData
from data.ParagraphExtractorTask import ParagraphExtractorTask
from data.ParagraphTranslations import ParagraphTranslations
from data.ParagraphsTranslations import ParagraphsTranslations
from data.TrainableEntityExtractionTask import TrainableEntityExtractionTask


class Extractor:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    LABELED_DATA = "labeled_data"
    PREDICTION_DATA = "prediction_data"

    DATA_TYPE = {
        LABELED_DATA: LabeledData,
        PREDICTION_DATA: PredictionData,
        PARAGRAPH_EXTRACTION_NAME: ParagraphExtractionData,
    }

    def __init__(self, extraction_identifier: ExtractionIdentifier, options: list[Option] = None, multi_value: bool = False):
        self.extraction_identifier = extraction_identifier
        self.multi_value = multi_value
        self.options = options
        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.pdf_metadata_extraction_db = client["pdf_metadata_extraction"]
        self.mongo_filter = {
            "run_name": self.extraction_identifier.run_name,
            "extraction_name": self.extraction_identifier.extraction_name,
        }

    def get_db_name(self, data_type: str) -> str:
        camel_case_str = self.DATA_TYPE[data_type].__name__
        snake_case_str = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_str).lower()
        return snake_case_str

    def get_data_from_db(self, data_type: str) -> list[LabeledData | PredictionData | ParagraphExtractionData]:
        data_list: list[LabeledData | PredictionData | ParagraphExtractionData] = []
        for document in self.pdf_metadata_extraction_db[self.get_db_name(data_type)].find(self.mongo_filter):
            data = self.DATA_TYPE[data_type](**document)
            if data_type != PARAGRAPH_EXTRACTION_NAME:
                for segment in data.xml_segments_boxes:
                    segment.page_width = segment.page_width if segment.page_width else data.page_width
                    segment.page_height = segment.page_height if segment.page_height else data.page_height

            data_list.append(data)

        return data_list

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
        extraction_data: ExtractionData = self.get_extraction_data_for_training(self.get_data_from_db("labeled_data"))
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
        self.pdf_metadata_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def insert_suggestions_in_db(self, suggestions: list[Suggestion]) -> (bool, str):
        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_metadata_extraction_db.suggestions.insert_many([x.to_dict() for x in suggestions])
        xml_folder_path = XmlFile(extraction_identifier=self.extraction_identifier, to_train=False).xml_folder_path
        for suggestion in suggestions:
            entity_name = {"entity_name": suggestion.entity_name, "xml_file_name": ""}
            xml_file_name = {"xml_file_name": suggestion.xml_file_name, "entity_name": ""}
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **entity_name})
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **xml_file_name})
            path = Path(join(xml_folder_path, suggestion.xml_file_name))
            if not path.is_dir():
                path.unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        prediction_samples = self.get_prediction_samples(self.get_data_from_db(self.PREDICTION_DATA))
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.predict(prediction_samples)

    def get_paragraphs(self, task: ParagraphExtractorTask) -> ParagraphsTranslations:
        languages = [x.language for x in task.xmls]
        if not languages:
            return ParagraphsTranslations(key=task.key)

        main_languages = [x.language for x in task.xmls if x.is_main_language]
        main_language = main_languages[0] if main_languages else languages[0]
        data = self.get_data_from_db(PARAGRAPH_EXTRACTION_NAME)

        if not data:
            return ParagraphsTranslations(key=task.key, available_languages=languages, main_language=main_language)

        paragraph_extraction_task = data[0]
        paragraphs_from_languages: list[ParagraphsFromLanguage] = list()
        for xml_data in paragraph_extraction_task.xmls:
            segmentation_data = SegmentationData(
                page_width=0, page_height=0, xml_segments_boxes=xml_data.xml_segments_boxes, label_segments_boxes=[]
            )
            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier, to_train=True, xml_file_name=xml_data.xml_file_name
            )
            pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
            paragraphs_from_language = ParagraphsFromLanguage(
                language=xml_data.language,
                paragraphs=[ParagraphFeatures.from_pdf_data(pdf_data, x) for x in pdf_data.pdf_data_segments],
                is_main_language=xml_data.is_main_language,
            )
            paragraphs_from_languages.append(paragraphs_from_language)

        aligner_use_case = MultilingualParagraphAlignerUseCase(self.extraction_identifier)
        aligner_use_case.align_languages(paragraphs_from_languages)

        paragraphs = [ParagraphTranslations.from_paragraphs_from_language(x) for x in paragraphs_from_languages]
        return ParagraphsTranslations(
            key=task.key, available_languages=languages, main_language=main_language, paragraphs=paragraphs
        )

    def insert_paragraphs_in_db(self, paragraphs):
        pass

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
    def calculate_task(task: TrainableEntityExtractionTask | ParagraphExtractorTask) -> (bool, str):
        if task.task == Extractor.CREATE_MODEL_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant,
                extraction_name=task.extraction_name,
                metadata=task.metadata,
                output_path=DATA_PATH,
            )

            Extractor.remove_old_models(extractor_identifier)

            if task.xmls.options:
                options = task.xmls.options
            else:
                options = extractor_identifier.get_options()

            multi_value = task.xmls.multi_value
            extractor = Extractor(extractor_identifier, options, multi_value)
            return extractor.create_models()

        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=task.tenant,
                extraction_name=task.extraction_name,
                metadata=task.metadata,
                output_path=DATA_PATH,
            )
            extractor = Extractor(extractor_identifier)
            suggestions = extractor.get_suggestions()
            return extractor.insert_suggestions_in_db(suggestions)

        if task.task == PARAGRAPH_EXTRACTION_NAME:
            extractor_identifier = ExtractionIdentifier(
                run_name=PARAGRAPH_EXTRACTION_NAME,
                extraction_name=task.key,
                output_path=DATA_PATH,
            )
            extractor = Extractor(extractor_identifier)
            paragraphs = extractor.get_paragraphs(task)
            return extractor.insert_paragraphs_in_db(paragraphs)

        return False, "Error"
