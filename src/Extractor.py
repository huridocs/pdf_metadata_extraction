import shutil
from os.path import join
from pathlib import Path
from time import time

import pymongo
from langcodes import standardize_tag

from config import config_logger, MONGO_PORT, MONGO_HOST
from data.LabeledData import LabeledData
from data.Option import Option

from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.Suggestion import Suggestion
from data.ExtractionTask import ExtractionTask
from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentsPages
from metadata_extraction.PdfMetadataExtractor import PdfMetadataExtractor
from metadata_extraction.PdfData import PdfData

from metadata_extraction.XmlFile import XmlFile
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class Extractor:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, tenant: str, extraction_id: str, options: list[Option] = None, multi_value: bool = False):
        self.tenant = tenant
        self.extraction_id = extraction_id
        self.multi_value = multi_value
        self.options = options

        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.pdf_metadata_extraction_db = client["pdf_metadata_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "id": self.extraction_id}

        self.pdfs_data: list[PdfData] = list()
        self.labeled_data: list[LabeledData] = list()
        self.predictions_data: list[PredictionData] = list()

    def get_labeled_data(self):
        labeled_data_list = []
        for document in self.pdf_metadata_extraction_db.labeled_data.find(self.mongo_filter):
            labeled_data_list.append(LabeledData(**document))

        return labeled_data_list

    # move to filter_valid_segments_pages
    def get_page_numbers_to_keep(self, labeled_data_list):
        filter_valid_segment_pages = FilterValidSegmentsPages(self.tenant, self.extraction_id)
        if self.options:
            page_numbers_list: list[list[int]] = filter_valid_segment_pages.for_training(labeled_data_list)
        else:
            page_numbers_list = [list() for _ in labeled_data_list]

        config_logger.info(f"Filter pages for training: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")
        return page_numbers_list

    def set_pdf_data_for_training(self):
        start = time()
        config_logger.info(f"Loading data to create model for {self.tenant} {self.extraction_id}")
        labeled_data_list = self.get_labeled_data()
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, self.get_page_numbers_to_keep(labeled_data_list)):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                tenant=labeled_data.tenant,
                extraction_id=labeled_data.id,
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
        config_logger.info(f"Loading data to calculate suggestions for {self.tenant} {self.extraction_id}")
        prediction_data_list = []
        for document in self.pdf_metadata_extraction_db.prediction_data.find(self.mongo_filter):
            prediction_data_list.append(PredictionData(**document))

        filter_valid_pages = FilterValidSegmentsPages(self.tenant, self.extraction_id)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        config_logger.info(f"Filter pages for prediction: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            self.predictions_data.append(prediction_data)
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)

            xml_file = XmlFile(
                tenant=self.tenant,
                extraction_id=self.extraction_id,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )
            self.pdfs_data.append(PdfData.from_xml_file(xml_file, segmentation_data, page_numbers))

    def create_models(self):
        self.set_pdf_data_for_training()
        is_multi_option = len(self.options) > 1
        if is_multi_option:
            multi_option_extractor = MultiOptionExtractor(tenant=self.tenant, extraction_id=self.extraction_id)
            model_created = multi_option_extractor.create_model(self.get_multi_option_data())
        else:
            pdf_metadata_extractor = PdfMetadataExtractor(
                tenant=self.tenant, extraction_id=self.extraction_id, pdfs_data=self.pdfs_data
            )
            model_created = pdf_metadata_extractor.create_model(self.labeled_data)

        self.delete_training_data()
        return model_created

    def delete_training_data(self):
        training_xml_path = XmlFile.get_xml_folder_path(tenant=self.tenant, extraction_id=self.extraction_id, to_train=True)
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.pdf_metadata_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def insert_suggestions_in_db(self, suggestions: list[Suggestion]) -> (bool, str):
        if not suggestions:
            return False, "No data to calculate suggestions"

        config_logger.info(f"Calculated and inserting {len(suggestions)} suggestions")

        self.pdf_metadata_extraction_db.suggestions.insert_many([x.to_dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(self.tenant, self.extraction_id, False)
        for suggestion in suggestions:
            xml_name = {"xml_file_name": suggestion.xml_file_name}
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **xml_name})
            Path(join(xml_folder_path, suggestion.xml_file_name)).unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        self.set_pdf_data_for_predictions()

        suggestions = []
        for prediction_data in self.predictions_data:
            suggestions.append(Suggestion.get_empty(self.tenant, self.extraction_id, prediction_data.xml_file_name))

        if MultiOptionExtractor.exist_model(self.tenant, self.extraction_id):
            multi_option_extractor = MultiOptionExtractor(self.tenant, self.extraction_id)
            multi_option_predictions = multi_option_extractor.get_multi_option_predictions(self.pdfs_data)
            for multi_option_sample, suggestion in zip(multi_option_predictions, self.predictions_data):
                suggestion.add_prediction_multi_option(multi_option_sample.values)
        else:
            pdf_metadata_extractor = PdfMetadataExtractor(self.tenant, self.extraction_id, self.pdfs_data)
            semantic_predictions_texts = pdf_metadata_extractor.get_metadata_predictions()

            for i, suggestion in enumerate(suggestions):
                suggestion.add_segments(self.pdfs_data[i])
                suggestion.add_prediction(semantic_predictions_texts[i])

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

        return MultiOptionData(samples=multi_option_samples, options=self.options, multi_value=self.multi_value)

    @staticmethod
    def calculate_task(extraction_task: ExtractionTask) -> (bool, str):
        tenant = extraction_task.tenant
        extraction_id = extraction_task.params.id

        if extraction_task.task == Extractor.CREATE_MODEL_TASK_NAME:
            options = extraction_task.params.options
            multi_value = extraction_task.params.multi_value
            metadata_extraction = Extractor(tenant, extraction_id, options, multi_value)
            return metadata_extraction.create_models()

        if extraction_task.task == Extractor.SUGGESTIONS_TASK_NAME:
            config_logger.info("Calculating suggestions")
            metadata_extraction = Extractor(tenant, extraction_id)
            suggestions = metadata_extraction.get_suggestions()
            return metadata_extraction.insert_suggestions_in_db(suggestions)

        return False, "Error"
