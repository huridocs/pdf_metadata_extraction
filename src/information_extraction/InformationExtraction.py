import os
import shutil
from pathlib import Path
from time import sleep
from typing import List

import pymongo

from ServiceConfig import ServiceConfig
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from data.InformationExtractionTask import InformationExtractionTask
import lightgbm as lgb

from information_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from information_extraction.PdfFeatures.PdfSegment import PdfSegment
from information_extraction.XmlFile import XmlFile
from information_extraction.methods.lightgbm_stack_not_complementary_models.LightgbmStackNotComplementaryModels import (
    LightgbmStackNotComplementaryModels,
)

from semantic_information_extraction.SemanticInformationExtraction import (
    SemanticInformationExtraction,
)


class InformationExtraction:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.semantic_information_extraction = SemanticInformationExtraction(tenant, property_name)
        service_config = ServiceConfig()
        self.pdf_features: List[PdfFeatures] = list()
        self.model_path = f"{service_config.docker_volume_path}/{tenant}/{property_name}/segment_predictor_model/model.model"
        self.model = None
        self.load_model()
        client = pymongo.MongoClient(f"mongodb://{service_config.mongo_host}:{service_config.mongo_port}")
        self.pdf_information_extraction_db = client["pdf_information_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "property_name": self.property_name}
        self.segment_selector = LightgbmStackNotComplementaryModels()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)

    def set_segments_for_training(self):
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_information_extraction_db = client["pdf_information_extraction"]

        self.pdf_features = list()
        for document in pdf_information_extraction_db.labeleddata.find(self.mongo_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document)
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
        self.set_segments_for_training()

        if not len(self.pdf_features) or not sum([len(pdf_features.pdf_segments) for pdf_features in self.pdf_features]):
            return False, "No labeled data to create model"

        self.model = self.segment_selector.create_model(self.pdf_features)

        if not os.path.exists(Path(self.model_path).parents[0]):
            os.makedirs(Path(self.model_path).parents[0])

        self.model.save_model(self.model_path, num_iteration=self.model.best_iteration)
        sleep(3)

        self.create_semantic_model()
        shutil.rmtree(
            XmlFile.get_xml_folder_path(tenant=self.tenant, property_name=self.property_name, to_train=True),
            ignore_errors=True,
        )
        self.pdf_information_extraction_db.labeleddata.delete_many(self.mongo_filter)
        return True, ""

    def create_semantic_model(self):
        self.semantic_information_extraction.remove_models()
        semantic_extraction_data: List[SemanticExtractionData] = list()
        for document in self.pdf_information_extraction_db.labeleddata.find(self.mongo_filter, no_cursor_timeout=True):
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
            suggestion = self.get_suggested_segment(xml_file_name=labeled_data.xml_file_name, pdf_features=pdf_features)
            semantic_extraction_data.append(
                SemanticExtractionData(
                    text=labeled_data.label_text,
                    segment_text=suggestion.segment_text,
                    language_iso=labeled_data.language_iso,
                )
            )

        if len(semantic_extraction_data) < 7:
            return

        self.semantic_information_extraction.create_model(semantic_extraction_data)

    def get_suggestions(self) -> (bool, str):
        if not self.model:
            return False, "No model"

        suggestions = self.calculate_suggestions()

        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_information_extraction_db.suggestions.insert_many([x.dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(self.tenant, self.property_name, False)
        for suggestion in suggestions:
            self.pdf_information_extraction_db.predictiondata.delete_many({"xml_file_name": suggestion.xml_file_name})
            if os.path.exists(f"{xml_folder_path}/{suggestion.xml_file_name}"):
                os.remove(f"{xml_folder_path}/{suggestion.xml_file_name}")

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
            suggestions.append(
                self.get_suggested_segment(xml_file_name=prediction_data.xml_file_name, pdf_features=pdf_features)
            )

        segments_text = [x.segment_text for x in suggestions]
        texts = self.semantic_information_extraction.get_semantic_predictions(segments_text)

        for index, suggestion in enumerate(suggestions):
            suggestion.text = texts[index]

        return suggestions

    def get_suggested_segment(self, xml_file_name: str, pdf_features: PdfFeatures):
        predictions = self.segment_selector.predict(self.model, [pdf_features])
        predicted_segments: List[PdfSegment] = list()
        for index, segment in enumerate(pdf_features.pdf_segments):
            if predictions[index] > 0.5:
                predicted_segments.append(segment)

        segment_text = " ".join([x.text_content for x in predicted_segments])
        segment_boxes = [x.get_segment_box() for x in predicted_segments]
        segment_boxes = [x.correct_output_data_scale() for x in segment_boxes]
        return Suggestion(
            tenant=self.tenant,
            property_name=self.property_name,
            xml_file_name=xml_file_name,
            text=segment_text,
            segment_text=segment_text,
            page_number=predicted_segments[0].page_number if len(predicted_segments) else 1,
            segments_boxes=segment_boxes,
        )

    @staticmethod
    def calculate_task(
        information_extraction_task: InformationExtractionTask,
    ) -> (bool, str):
        if information_extraction_task.task == InformationExtraction.CREATE_MODEL_TASK_NAME:
            information_extraction = InformationExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.create_models()

        if information_extraction_task.task == InformationExtraction.SUGGESTIONS_TASK_NAME:
            information_extraction = InformationExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.get_suggestions()

        return False, "Error"
