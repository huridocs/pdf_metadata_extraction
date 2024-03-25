import os
import shutil
from os.path import join, exists
from pathlib import Path
from time import time

import pymongo
from langcodes import standardize_tag

from config import config_logger, MONGO_PORT, MONGO_HOST, DATA_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option

from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.Suggestion import Suggestion
from data.ExtractionTask import ExtractionTask
from FilterValidSegmentsPages import FilterValidSegmentsPages
from metadata_extraction.PdfMetadataExtractor import PdfMetadataExtractor
from metadata_extraction.PdfData import PdfData

from metadata_extraction.XmlFile import XmlFile
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class Extractor:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, extraction_identifier: ExtractionIdentifier, options: list[Option] = None, multi_value: bool = False):
        self.extraction_identifier = extraction_identifier
        self.multi_value = multi_value
        self.options = options

        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.pdf_metadata_extraction_db = client["pdf_metadata_extraction"]
        self.mongo_filter = {"tenant": self.extraction_identifier.run_name, "id": self.extraction_identifier.extraction_name}

        self.pdfs_data: list[PdfData] = list()
        self.labeled_data: list[LabeledData] = list()
        self.predictions_data: list[PredictionData] = list()

    def get_labeled_data(self):
        labeled_data_list = []
        for document in self.pdf_metadata_extraction_db.labeled_data.find(self.mongo_filter):
            labeled_data_list.append(LabeledData(**document))

        return labeled_data_list

    def set_pdf_data_for_training(self):
        start = time()
        config_logger.info(f"Loading data to create model for {str(self.extraction_identifier)}")
        labeled_data_list = self.get_labeled_data()
        page_numbers_list = FilterValidSegmentsPages(self.extraction_identifier).for_training(labeled_data_list)
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            extraction_identifier = ExtractionIdentifier(run_name=labeled_data.tenant, extraction_name=labeled_data.id)
            xml_file = XmlFile(
                extraction_identifier=extraction_identifier,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            pdf_segments = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers_to_keep)

            if not pdf_segments:
                continue

            self.labeled_data.append(labeled_data)
            self.pdfs_data.append(pdf_segments)

        config_logger.info(f"Set pdf data {round(time() - start, 2)} seconds")

    def set_pdf_data_for_predictions(self):
        config_logger.info(f"Loading data to calculate suggestions for {self.extraction_identifier}")
        prediction_data_list = []
        for document in self.pdf_metadata_extraction_db.prediction_data.find(self.mongo_filter):
            prediction_data_list.append(PredictionData(**document))

        filter_valid_pages = FilterValidSegmentsPages(self.extraction_identifier)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        config_logger.info(f"Filter pages for prediction: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            self.predictions_data.append(prediction_data)
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)

            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )
            self.pdfs_data.append(PdfData.from_xml_file(xml_file, segmentation_data, page_numbers))

    def create_models(self):
        self.set_pdf_data_for_training()
        is_multi_option = len(self.options) > 1
        if is_multi_option:
            multi_option_extractor = MultiOptionExtractor(self.extraction_identifier)
            model_created = multi_option_extractor.create_model(self.get_multi_option_data())
        else:
            pdf_metadata_extractor = PdfMetadataExtractor(
                extraction_identifier=self.extraction_identifier, pdfs_data=self.pdfs_data
            )
            model_created = pdf_metadata_extractor.create_model(self.labeled_data)

        self.delete_training_data()
        return model_created

    def delete_training_data(self):
        training_xml_path = XmlFile.get_xml_folder_path(extraction_identifier=self.extraction_identifier, to_train=True)
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.pdf_metadata_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def insert_suggestions_in_db(self, suggestions: list[Suggestion]) -> (bool, str):
        if not suggestions:
            return False, "No data to calculate suggestions"

        config_logger.info(f"Calculated and inserting {len(suggestions)} suggestions")

        self.pdf_metadata_extraction_db.suggestions.insert_many([x.to_dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(extraction_identifier=self.extraction_identifier, to_train=False)
        for suggestion in suggestions:
            xml_name = {"xml_file_name": suggestion.xml_file_name}
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **xml_name})
            Path(join(xml_folder_path, suggestion.xml_file_name)).unlink(missing_ok=True)

        return True, ""

    def get_multi_option_suggestions(self):
        suggestions = self.get_empty_suggestions()
        multi_option_extractor = MultiOptionExtractor(self.extraction_identifier)
        multi_option_predictions = multi_option_extractor.get_multi_option_predictions(self.pdfs_data)
        for suggestion, multi_option_sample in zip(suggestions, multi_option_predictions):
            suggestion.add_prediction_multi_option(multi_option_sample)

        return suggestions

    def get_suggestions(self) -> list[Suggestion]:
        self.set_pdf_data_for_predictions()

        if MultiOptionExtractor.is_multi_option_extraction(self.extraction_identifier):
            return self.get_multi_option_suggestions()

        pdf_metadata_extractor = PdfMetadataExtractor(self.extraction_identifier, self.pdfs_data)
        semantic_predictions_texts = pdf_metadata_extractor.get_metadata_predictions()

        if not semantic_predictions_texts:
            return []

        suggestions = self.get_empty_suggestions()

        for suggestion, semantic_prediction, pdf_data in zip(suggestions, semantic_predictions_texts, self.pdfs_data):
            suggestion.add_prediction(semantic_prediction, pdf_data)

        return suggestions

    def get_empty_suggestions(self):
        suggestions = []
        for prediction_data in self.predictions_data:
            suggestions.append(Suggestion.get_empty(self.extraction_identifier, prediction_data.xml_file_name))
        return suggestions

    def get_multi_option_data(self):
        multi_option_samples: list[MultiOptionSample] = list()
        for pdf_data, labeled_data in zip(self.pdfs_data, self.labeled_data):
            multi_option_sample = MultiOptionSample(
                pdf_data=pdf_data,
                values=labeled_data.values,
                language_iso=standardize_tag(labeled_data.language_iso),
            )
            multi_option_samples.append(multi_option_sample)

        return MultiOptionData(
            samples=multi_option_samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

    @staticmethod
    def remove_old_models(extractor_identifier: ExtractionIdentifier):
        if exists(extractor_identifier.get_path()):
            os.utime(extractor_identifier.get_path())

        for run_name in os.listdir(DATA_PATH):
            if run_name == "cache":
                continue

            for extraction_name in os.listdir(join(DATA_PATH, run_name)):
                extractor_identifier_to_check = ExtractionIdentifier(run_name=run_name, extraction_name=extraction_name)
                if extractor_identifier_to_check.is_old():
                    shutil.rmtree(extractor_identifier_to_check.get_path(), ignore_errors=True)

    @staticmethod
    def calculate_task(extraction_task: ExtractionTask) -> (bool, str):
        extraction_name = extraction_task.params.id
        extractor_identifier = ExtractionIdentifier(run_name=extraction_task.tenant, extraction_name=extraction_name)
        Extractor.remove_old_models(extractor_identifier)

        if extraction_task.task == Extractor.CREATE_MODEL_TASK_NAME:
            options = extraction_task.params.options
            multi_value = extraction_task.params.multi_value
            extractor = Extractor(extractor_identifier, options, multi_value)
            return extractor.create_models()

        if extraction_task.task == Extractor.SUGGESTIONS_TASK_NAME:
            config_logger.info("Calculating suggestions")
            extractor = Extractor(extractor_identifier)
            suggestions = extractor.get_suggestions()
            return extractor.insert_suggestions_in_db(suggestions)

        return False, "Error"
