import logging
import shutil
from os.path import join
from pathlib import Path
from time import time
from typing import List

import pymongo
from langcodes import standardize_tag

from ServiceConfig import ServiceConfig
from config import config_logger
from data.LabeledData import LabeledData
from data.Option import Option
from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from data.MetadataExtractionTask import MetadataExtractionTask
from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentPages

from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.XmlFile import XmlFile
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor
from segment_selector.SegmentSelector import SegmentSelector
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction


class MetadataExtraction:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, tenant: str, property_name: str, multi_option: bool, logger: logging.Logger = None):
        self.tenant = tenant
        self.property_name = property_name
        self.multi_option = multi_option

        service_config = ServiceConfig()
        self.filter_valid_pages = FilterValidSegmentPages(tenant, property_name)
        
        client = pymongo.MongoClient(f"mongodb://{service_config.mongo_host}:{service_config.mongo_port}")
        self.pdf_information_extraction_db = client["pdf_information_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "property_name": self.property_name}

        self.pdf_features: List[PdfFeatures] = list()
        self.labeled_data: List[LabeledData] = list()
        self.multilingual: bool = False

    def set_pdf_features_for_training(self):
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_information_extraction_db = client["pdf_information_extraction"]

        self.multilingual = False
        self.pdf_features = list()
        self.labeled_data = list()

        collection = pdf_information_extraction_db.labeled_data
        labeled_data_list = []
        for document in collection.find(self.mongo_filter, no_cursor_timeout=True):
            labeled_data_list.append(LabeledData(**document))

        page_numbers_list = self.filter_valid_pages.for_training(labeled_data_list)
        config_logger.info(f"Filter pages for training: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        for labeled_data, page_numbers in zip(labeled_data_list, page_numbers_list):
            if labeled_data.language_iso != "en" and labeled_data.language_iso != "eng":
                self.multilingual = True

            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                tenant=labeled_data.tenant,
                property_name=labeled_data.property_name,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            pdf_features = PdfFeatures.from_xml_file(xml_file, segmentation_data, page_numbers)

            if not pdf_features:
                continue

            self.labeled_data.append(labeled_data)
            self.pdf_features.append(pdf_features)

    def create_models(self, options: List[Option], multi_value: bool = False):
        config_logger.info(f"Loading data to create model for {self.tenant} {self.property_name}")
        self.set_pdf_features_for_training()

        if not len(self.pdf_features) or not sum([len(pdf_features.pdf_segments) for pdf_features in self.pdf_features]):
            self.delete_training_data()
            return False, "No labeled data to create model"

        start = time()
        config_logger.info(f"Creating model with {len(self.pdf_features)} documents for {self.tenant} {self.property_name}")
        segment_selector = SegmentSelector(tenant=self.tenant, property_name=self.property_name)
        segment_selector.create_model(pdfs_features=self.pdf_features, multilingual=self.multilingual)

        config_logger.info(f"Finished creating model {int(time() - start)} seconds")

        config_logger.info(f"Creating semantic model")
        start = time()
        if self.multi_option:
            multi_option_extractor = MultiOptionExtractor(tenant=self.tenant, property_name=self.property_name)
            multi_option_extractor.create_model(self.get_multi_option_extraction_data(options, multi_value))
        else:
            self.create_semantic_model()

        config_logger.info(f"Finished semantic model in {int(time() - start)} seconds")

        self.delete_training_data()
        return True, ""

    def delete_training_data(self):
        training_xml_path = XmlFile.get_xml_folder_path(tenant=self.tenant, property_name=self.property_name, to_train=True)
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.pdf_information_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def create_semantic_model(self):
        semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)
        semantic_metadata_extraction.remove_models()
        semantic_extraction_data: List[SemanticExtractionData] = list()
        for pdf_features, labeled_data in zip(self.pdf_features, self.labeled_data):
            pdf_segments = [pdf_segment.text_content for pdf_segment in pdf_features.pdf_segments if pdf_segment.ml_label]
            suggestion_text = " ".join(pdf_segments)
            semantic_extraction_data.append(
                SemanticExtractionData(
                    text=labeled_data.label_text.strip(),
                    segment_text=suggestion_text,
                    language_iso=standardize_tag(labeled_data.language_iso),
                )
            )

        semantic_metadata_extraction.create_model(semantic_extraction_data)

    def insert_suggestions_in_db(self) -> (bool, str):
        segment_selector = SegmentSelector(self.tenant, self.property_name)
        if not segment_selector.model:
            return False, "No model"

        suggestions = self.get_suggestions()

        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_information_extraction_db.suggestions.insert_many([x.dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(self.tenant, self.property_name, False)
        for suggestion in suggestions:
            self.pdf_information_extraction_db.predictiondata.delete_many({"xml_file_name": suggestion.xml_file_name})
            Path(join(xml_folder_path, suggestion.xml_file_name)).unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self):
        suggestions: List[Suggestion] = self.get_segment_selector_suggestions()

        segments_text = [x.segment_text for x in suggestions]
        if self.multi_option:
            multi_option_extraction = MultiOptionExtractor(self.tenant, self.property_name)
            multi_option_prediction = multi_option_extraction.get_multi_option_predictions(segments_text)
            for index, suggestion in enumerate(suggestions):
                suggestion.options = multi_option_prediction[index].options
        else:
            semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)
            texts = semantic_metadata_extraction.get_semantic_predictions(segments_text)

            for index, suggestion in enumerate(suggestions):
                suggestion.text = texts[index]

        return suggestions

    def get_segment_selector_suggestions(self):
        suggestions: List[Suggestion] = list()
        predictions_data: List[PredictionData] = list()
        pdfs_features: List[PdfFeatures] = list()

        config_logger.info(f"Loading data to calculate suggestions for {self.tenant} {self.property_name}")

        prediction_data_list = []

        for document in self.pdf_information_extraction_db.predictiondata.find(self.mongo_filter, no_cursor_timeout=True):
            prediction_data_list.append(PredictionData(**document))

        page_numbers_list = self.filter_valid_pages.for_prediction(prediction_data_list)
        config_logger.info(f"Filter pages for prediction: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            predictions_data.append(prediction_data)
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)

            xml_file = XmlFile(
                tenant=self.tenant,
                property_name=self.property_name,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )
            pdfs_features.append(PdfFeatures.from_xml_file(xml_file, segmentation_data, page_numbers))

        config_logger.info(f"Calculating {len(predictions_data)} suggestions for {self.tenant} {self.property_name}")

        for prediction_data, pdf_features in zip(predictions_data, pdfs_features):
            suggestion = self.get_suggested_segment(xml_file_name=prediction_data.xml_file_name, pdf_features=pdf_features)
            suggestions.append(suggestion)

        return suggestions

    def get_suggested_segment(self, xml_file_name: str, pdf_features: PdfFeatures):
        if not pdf_features.pdf_segments:
            return Suggestion(
                tenant=self.tenant,
                property_name=self.property_name,
                xml_file_name=xml_file_name,
                text="",
                segment_text="",
                page_number=1,
                segments_boxes=[],
            )

        segment_selector = SegmentSelector(self.tenant, self.property_name)
        segment_selector.set_extraction_segments(pdfs_features=[pdf_features])

        extraction_pdf_segments = [x for x in pdf_features.pdf_segments if x.ml_label]
        segment_text = " ".join([pdf_segment.text_content for pdf_segment in extraction_pdf_segments])
        segment_boxes = [pdf_segment.get_segment_box() for pdf_segment in extraction_pdf_segments]
        segment_boxes = [segment_box.correct_output_data_scale() for segment_box in segment_boxes]

        return Suggestion(
            tenant=self.tenant,
            property_name=self.property_name,
            xml_file_name=xml_file_name,
            text=segment_text,
            segment_text=segment_text,
            page_number=extraction_pdf_segments[0].page_number if len(extraction_pdf_segments) else 1,
            segments_boxes=segment_boxes,
        )

    def get_multi_option_extraction_data(self, options, multi_value):
        samples: List[MultiOptionExtractionSample] = list()
        for pdf_features, labeled_data in zip(self.pdf_features, self.labeled_data):
            pdf_segments = [pdf_segment.text_content for pdf_segment in pdf_features.pdf_segments if pdf_segment.ml_label]
            suggestion_text = " ".join(pdf_segments)
            samples.append(MultiOptionExtractionSample(text=suggestion_text, options=labeled_data.options))

        return MultiOptionExtractionData(
            multilingual=self.multilingual, multi_value=multi_value, options=options, samples=samples
        )

    @staticmethod
    def calculate_task(information_extraction_task: MetadataExtractionTask, logger: logging.Logger = None) -> (bool, str):
        tenant = information_extraction_task.tenant
        property_name = information_extraction_task.params.property_name

        if information_extraction_task.task == MetadataExtraction.CREATE_MODEL_TASK_NAME:
            multi_option = True if information_extraction_task.params.options else False
            metadata_extraction = MetadataExtraction(tenant, property_name, multi_option, logger)
            return metadata_extraction.create_models(
                options=information_extraction_task.params.options,
                multi_value=information_extraction_task.params.multi_value,
            )

        if information_extraction_task.task == MetadataExtraction.SUGGESTIONS_TASK_NAME:
            multi_option = MultiOptionExtractor.exist_model(tenant, property_name)
            metadata_extraction = MetadataExtraction(tenant, property_name, multi_option, logger)
            return metadata_extraction.insert_suggestions_in_db()

        return False, "Error"
