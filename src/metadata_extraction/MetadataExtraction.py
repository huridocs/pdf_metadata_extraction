import os
import shutil
from os.path import exists
from typing import List, Optional

import pymongo

from ServiceConfig import ServiceConfig
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from data.InformationExtractionTask import InformationExtractionTask

from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.XmlFile import XmlFile
from segment_selector.SegmentSelector import SegmentSelector
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction


class MetadataExtraction:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name

        service_config = ServiceConfig()
        client = pymongo.MongoClient(f"mongodb://{service_config.mongo_host}:{service_config.mongo_port}")
        self.pdf_information_extraction_db = client["pdf_information_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "property_name": self.property_name}

        self.pdf_features: List[PdfFeatures] = list()
        self.multilingual: bool = False
        self.semantic_information_extraction = SemanticMetadataExtraction(tenant, property_name)
        self.segment_selector: Optional[SegmentSelector] = None

    def set_pdf_features_for_training(self):
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_information_extraction_db = client["pdf_information_extraction"]
        self.multilingual = False
        self.pdf_features = list()
        for document in pdf_information_extraction_db.labeled_data.find(self.mongo_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document)

            if labeled_data.language_iso != "en" and labeled_data.language_iso != "eng":
                self.multilingual = True

            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                tenant=labeled_data.tenant,
                property_name=labeled_data.property_name,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )
            pdf_features = PdfFeatures.from_xml_file(xml_file, segmentation_data)
            if pdf_features:
                self.pdf_features.append(pdf_features)

    def create_models(self):
        self.set_pdf_features_for_training()

        if not len(self.pdf_features) or not sum([len(pdf_features.pdf_segments) for pdf_features in self.pdf_features]):
            return False, "No labeled data to create model"

        segment_selector = SegmentSelector(tenant=self.tenant, property_name=self.property_name)
        segment_selector.create_model(pdfs_features=self.pdf_features, multilingual=self.multilingual)

        self.create_semantic_model()
        shutil.rmtree(
            XmlFile.get_xml_folder_path(tenant=self.tenant, property_name=self.property_name, to_train=True),
            ignore_errors=True,
        )
        self.pdf_information_extraction_db.labeled_data.delete_many(self.mongo_filter)
        return True, ""

    def create_semantic_model(self):
        self.semantic_information_extraction.remove_models()
        semantic_extraction_data: List[SemanticExtractionData] = list()
        for document in self.pdf_information_extraction_db.labeled_data.find(self.mongo_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document)
            xml_file = XmlFile(
                tenant=self.tenant,
                property_name=self.property_name,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            pdf_features = PdfFeatures.from_xml_file(xml_file, SegmentationData.from_labeled_data(labeled_data))
            if not pdf_features.pdf_segments:
                continue

            pdf_segments = [pdf_segment.text_content for pdf_segment in pdf_features.pdf_segments if pdf_segment.ml_label]
            suggestion_text = " ".join(pdf_segments)
            semantic_extraction_data.append(
                SemanticExtractionData(
                    text=labeled_data.label_text,
                    segment_text=suggestion_text,
                    language_iso=labeled_data.language_iso,
                )
            )

        if len(semantic_extraction_data) < 7:
            return

        self.semantic_information_extraction.create_model(semantic_extraction_data)

    def get_suggestions(self) -> (bool, str):
        self.segment_selector = SegmentSelector(self.tenant, self.property_name)
        if not self.segment_selector.model:
            return False, "No model"

        suggestions = self.calculate_suggestions()

        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_information_extraction_db.suggestions.insert_many([x.dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(self.tenant, self.property_name, False)
        for suggestion in suggestions:
            self.pdf_information_extraction_db.predictiondata.delete_many({"xml_file_name": suggestion.xml_file_name})
            suggestion_xml_path = os.path.join(xml_folder_path, suggestion.xml_file_name)
            if exists(suggestion_xml_path):
                os.remove(suggestion_xml_path)

        return True, ""

    def calculate_suggestions(self):
        suggestions: List[Suggestion] = list()

        for document in self.pdf_information_extraction_db.predictiondata.find(self.mongo_filter, no_cursor_timeout=True):
            prediction_data = PredictionData(**document)
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)

            xml_file = XmlFile(
                tenant=self.tenant,
                property_name=self.property_name,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )
            pdf_features = PdfFeatures.from_xml_file(xml_file, segmentation_data)

            if not pdf_features.pdf_segments:
                continue

            suggestion = self.get_suggested_segment(xml_file_name=prediction_data.xml_file_name, pdf_features=pdf_features)
            suggestions.append(suggestion)

        segments_text = [x.segment_text for x in suggestions]
        texts = self.semantic_information_extraction.get_semantic_predictions(segments_text)

        for index, suggestion in enumerate(suggestions):
            suggestion.text = texts[index]

        return suggestions

    def get_suggested_segment(self, xml_file_name: str, pdf_features: PdfFeatures):
        self.segment_selector.set_extraction_segments(pdfs_features=[pdf_features])

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

    @staticmethod
    def calculate_task(
        information_extraction_task: InformationExtractionTask,
    ) -> (bool, str):
        if information_extraction_task.task == MetadataExtraction.CREATE_MODEL_TASK_NAME:
            information_extraction = MetadataExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.create_models()

        if information_extraction_task.task == MetadataExtraction.SUGGESTIONS_TASK_NAME:
            information_extraction = MetadataExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.get_suggestions()

        return False, "Error"
