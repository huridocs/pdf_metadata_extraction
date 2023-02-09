import logging
import shutil
from os.path import join
from pathlib import Path
from time import time

import pymongo
from langcodes import standardize_tag

from config import config_logger, MONGO_PORT, MONGO_HOST
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfTagData import PdfTagData
from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from data.Suggestion import Suggestion
from data.MetadataExtractionTask import MetadataExtractionTask
from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentPages

from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.PdfFeatures.PdfSegment import PdfSegment
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

        self.filter_valid_pages = FilterValidSegmentPages(tenant, property_name)

        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.pdf_information_extraction_db = client["pdf_metadata_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "property_name": self.property_name}

        self.pdf_features: list[PdfFeatures] = list()
        self.labeled_data: list[LabeledData] = list()
        self.multilingual: bool = False

        self.segment_selector = SegmentSelector(self.tenant, self.property_name)

    def set_pdf_features_for_training(self):
        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        pdf_information_extraction_db = client["pdf_metadata_extraction"]

        self.multilingual = False
        self.pdf_features = list()
        self.labeled_data = list()

        collection = pdf_information_extraction_db.labeled_data
        labeled_data_list = []
        for document in collection.find(self.mongo_filter):
            labeled_data_list.append(LabeledData(**document))

        page_numbers_list = self.filter_valid_pages.for_training(labeled_data_list)
        config_logger.info(f"Filter pages for training: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        start = time()
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

        config_logger.info(f"PdfFeatures {round(time() - start, 2)} seconds")

    def create_models(self):
        start = time()
        config_logger.info(f"Loading data to create model for {self.tenant} {self.property_name}")
        self.set_pdf_features_for_training()
        print(f"set pdf features {round(time() - start, 2)} seconds")

        if not len(self.pdf_features) or not sum([len(pdf_features.pdf_segments) for pdf_features in self.pdf_features]):
            self.delete_training_data()
            return False, "No labeled data to create model"

        start = time()
        config_logger.info(f"Creating model with {len(self.pdf_features)} documents for {self.tenant} {self.property_name}")
        self.segment_selector.create_model(pdfs_features=self.pdf_features)

        config_logger.info(f"Finished creating model {round(time() - start, 2)} seconds")

        config_logger.info(f"Creating semantic model")
        start = time()
        self.create_semantic_model()

        config_logger.info(f"Finished semantic model in {round(time() - start, 2)} seconds")

        self.delete_training_data()
        return True, ""

    def delete_training_data(self):
        training_xml_path = XmlFile.get_xml_folder_path(tenant=self.tenant, property_name=self.property_name, to_train=True)
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.pdf_information_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def create_semantic_model(self):
        semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)
        semantic_metadata_extraction.remove_models()
        semantic_extractions_data: list[SemanticExtractionData] = list()
        for pdf_features, labeled_data in zip(self.pdf_features, self.labeled_data):
            semantic_extraction_data = SemanticExtractionData(
                text=labeled_data.label_text.strip(),
                pdf_tags=(self.get_predicted_tags_data(pdf_features)),
                language_iso=standardize_tag(labeled_data.language_iso),
            )

            semantic_extractions_data.append(semantic_extraction_data)

        semantic_metadata_extraction.create_model(semantic_extractions_data)

    @staticmethod
    def get_predicted_tags_data(pdf_features: PdfFeatures) -> list[PdfTagData]:
        all_tags = pdf_features.get_tags()

        predicted_pdf_segments = [x for x in pdf_features.pdf_segments if x.ml_label]

        pdf_tags = list()
        for pdf_segment in predicted_pdf_segments:
            for pdf_tag in [x for x in all_tags if x.page_number == pdf_segment.page_number]:
                if pdf_segment.is_selected(pdf_tag.bounding_box):
                    pdf_tags.append(pdf_tag)

        return [x.to_pdf_tag_data() for x in pdf_tags]

    def insert_suggestions_in_db(self) -> (bool, str):
        if not self.segment_selector.model:
            return False, "No model"

        suggestions = self.get_suggestions()

        if not suggestions:
            return False, "No data to calculate suggestions"

        config_logger.info(f"Calculated and inserting {len(suggestions)} suggestions")

        self.pdf_information_extraction_db.suggestions.insert_many([x.dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(self.tenant, self.property_name, False)
        for suggestion in suggestions:
            xml_name = {"xml_file_name": suggestion.xml_file_name}
            self.pdf_information_extraction_db.prediction_data.delete_many({**self.mongo_filter, **xml_name})
            Path(join(xml_folder_path, suggestion.xml_file_name)).unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        start = time()
        semantic_predictions_data, suggestions = self.get_segment_selector_prediction_data()
        config_logger.info(f"get_segment_selector_suggestions {round(time() - start, 2)} seconds")

        start = time()
        semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)
        semantic_predictions_texts = semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        for semantic_prediction_text, suggestion in zip(semantic_predictions_texts, suggestions):
            suggestion.add_prediction(semantic_prediction_text)

        config_logger.info(f"get_semantic_predictions {round(time() - start, 2)} seconds")
        return suggestions

    def get_segment_selector_prediction_data(self) -> (list[SemanticPredictionData], list[Suggestion]):
        predictions_data, pdfs_features = self.get_pdf_features()

        config_logger.info(f"Calculating {len(predictions_data)} suggestions for {self.tenant} {self.property_name}")

        semantic_predictions_data: list[SemanticPredictionData] = list()
        suggestions: list[Suggestion] = list()

        for prediction_data, pdf_features in zip(predictions_data, pdfs_features):
            suggestion = Suggestion.get_empty(self.tenant, self.property_name, prediction_data.xml_file_name)

            if not pdf_features.pdf_segments:
                semantic_predictions_data.append(SemanticPredictionData.from_text(""))
                suggestions.append(suggestion)
                continue

            self.segment_selector.set_extraction_segments(pdfs_features=[pdf_features])
            semantic_predictions_data.append(SemanticPredictionData(pdf_tags=self.get_predicted_tags_data(pdf_features)))
            suggestions.append(suggestion.add_segments(pdf_features))

        return semantic_predictions_data, suggestions

    def get_pdf_features(self):
        pdfs_features: list[PdfFeatures] = list()
        config_logger.info(f"Loading data to calculate suggestions for {self.tenant} {self.property_name}")
        prediction_data_list = []
        for document in self.pdf_information_extraction_db.prediction_data.find(self.mongo_filter):
            prediction_data_list.append(PredictionData(**document))

        page_numbers_list = self.filter_valid_pages.for_prediction(prediction_data_list)
        config_logger.info(f"Filter pages for prediction: total {len(page_numbers_list)} documents.")
        config_logger.info(f"Filter: {page_numbers_list}")

        predictions_data: list[PredictionData] = list()
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

        return predictions_data, pdfs_features

    def get_multi_option_extraction_data(self, options, multi_value):
        samples: list[MultiOptionExtractionSample] = list()
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
            return metadata_extraction.create_models()

        if information_extraction_task.task == MetadataExtraction.SUGGESTIONS_TASK_NAME:
            config_logger.info("Calculating suggestions")
            multi_option = MultiOptionExtractor.exist_model(tenant, property_name)
            metadata_extraction = MetadataExtraction(tenant, property_name, multi_option, logger)
            return metadata_extraction.insert_suggestions_in_db()

        return False, "Error"
