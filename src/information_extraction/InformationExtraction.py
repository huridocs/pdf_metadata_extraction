import os
import shutil
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import pymongo

from ServiceConfig import ServiceConfig
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.SegmentationData import SegmentationData
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from data.InformationExtractionTask import InformationExtractionTask
from information_extraction.Segment import Segment
import lightgbm as lgb

from information_extraction.XmlFile import XmlFile
from semantic_information_extraction.SemanticInformationExtraction import (
    SemanticInformationExtraction,
)


class InformationExtraction:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.semantic_information_extraction = SemanticInformationExtraction(
            tenant, property_name
        )
        service_config = ServiceConfig()
        self.segments: List[Segment] = list()
        self.model_path = f"{service_config.docker_volume_path}/{tenant}/{property_name}/segment_predictor_model/model.model"
        self.model = None
        self.load_model()
        client = pymongo.MongoClient(
            f"mongodb://{service_config.mongo_host}:{service_config.mongo_port}"
        )
        self.pdf_information_extraction_db = client["pdf_information_extraction"]
        self.mongo_filter = {"tenant": self.tenant, "property_name": self.property_name}

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)

    def set_segments_for_training(self):
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_information_extraction_db = client["pdf_information_extraction"]

        self.segments = []
        for document in pdf_information_extraction_db.labeleddata.find(
            self.mongo_filter, no_cursor_timeout=True
        ):
            labeled_data = LabeledData(**document)
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                tenant=labeled_data.tenant,
                property_name=labeled_data.property_name,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )
            self.segments.extend(xml_file.get_segments(segmentation_data))

    def create_models(self):
        self.set_segments_for_training()

        if len(self.segments) == 0:
            return False, "No labeled data to create model"

        self.run_light_gbm()
        self.create_semantic_model()
        shutil.rmtree(
            XmlFile.get_xml_folder_path(
                tenant=self.tenant, property_name=self.property_name, to_train=True
            ),
            ignore_errors=True,
        )
        self.pdf_information_extraction_db.labeleddata.delete_many(self.mongo_filter)
        return True, ""

    def create_semantic_model(self):
        self.semantic_information_extraction.remove_models()
        semantic_extraction_data: List[SemanticExtractionData] = list()
        for document in self.pdf_information_extraction_db.labeleddata.find(
            self.mongo_filter, no_cursor_timeout=True
        ):
            labeled_data = LabeledData(**document)
            xml_file = XmlFile(
                tenant=self.tenant,
                property_name=self.property_name,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            self.segments = xml_file.get_segments(
                SegmentationData.from_labeled_data(labeled_data)
            )
            suggestion = self.get_suggested_segment(
                xml_file_name=labeled_data.xml_file_name
            )
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

    def run_light_gbm(self):
        x_train, y_train = self.get_training_data()

        if x_train is None:
            return

        parameters = dict()
        parameters["num_leaves"] = 35
        parameters["feature_fraction"] = 1
        parameters["bagging_fraction"] = 1
        parameters["bagging_freq"] = 0
        parameters["objective"] = "binary"
        parameters["learning_rate"] = 0.05
        parameters["metric"] = "binary_logloss"
        parameters["verbose"] = -1
        parameters["boosting_type"] = "gbdt"

        train_data = lgb.Dataset(x_train, y_train)
        num_round = 3000
        light_gbm_model = lgb.train(parameters, train_data, num_round)

        if not light_gbm_model:
            return

        self.model = light_gbm_model
        if not os.path.exists(Path(self.model_path).parents[0]):
            os.makedirs(Path(self.model_path).parents[0])

        self.model.save_model(self.model_path, num_iteration=self.model.best_iteration)
        sleep(3)

    def get_suggestions(self) -> (bool, str):
        if not self.model:
            return False, "No model"

        suggestions = self.calculate_suggestions()

        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_information_extraction_db.suggestions.insert_many(
            [x.dict() for x in suggestions]
        )
        xml_folder_path = XmlFile.get_xml_folder_path(
            self.tenant, self.property_name, False
        )
        for suggestion in suggestions:
            self.pdf_information_extraction_db.predictiondata.delete_many(
                {"xml_file_name": suggestion.xml_file_name}
            )
            if os.path.exists(f"{xml_folder_path}/{suggestion.xml_file_name}"):
                os.remove(f"{xml_folder_path}/{suggestion.xml_file_name}")

        return True, ""

    def calculate_suggestions(self):
        suggestions: List[Suggestion] = list()

        for document in self.pdf_information_extraction_db.predictiondata.find(
            self.mongo_filter, no_cursor_timeout=True
        ):
            prediction_data = PredictionData(**document)
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)

            xml_file = XmlFile(
                tenant=self.tenant,
                property_name=self.property_name,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )
            self.segments = xml_file.get_segments(segmentation_data)
            suggestions.append(
                self.get_suggested_segment(xml_file_name=prediction_data.xml_file_name)
            )

        segments_text = [x.segment_text for x in suggestions]
        texts = self.semantic_information_extraction.get_semantic_predictions(
            segments_text
        )

        for index, suggestion in enumerate(suggestions):
            suggestion.text = texts[index]

        return suggestions

    def get_suggested_segment(self, xml_file_name: str):
        if not self.segments:
            return Suggestion(
                tenant=self.tenant,
                property_name=self.property_name,
                xml_file_name=xml_file_name,
                text="",
                segment_text="",
                page_number=1,
            )

        x, y = self.get_training_data()
        predictions = self.model.predict(x)
        predicted_segments: List[Segment] = list()
        for index, segment in enumerate(self.segments):
            if predictions[index] > 0.5:
                predicted_segments.append(segment)
        segment_text = " ".join([x.text_content for x in predicted_segments])
        return Suggestion(
            tenant=self.tenant,
            property_name=self.property_name,
            xml_file_name=xml_file_name,
            text=segment_text,
            segment_text=segment_text,
            page_number=predicted_segments[0].page_number
            if len(predicted_segments)
            else 1,
        )

    def get_training_data(self):
        X = None
        y = np.array([])

        for segment in self.segments:
            features = segment.get_features_array()
            features = features.reshape(1, len(features))
            X = features if X is None else np.concatenate((X, features), axis=0)
            y = np.append(y, segment.ml_class_label)

        return X, y

    @staticmethod
    def calculate_task(
        information_extraction_task: InformationExtractionTask,
    ) -> (bool, str):
        if (
            information_extraction_task.task
            == InformationExtraction.CREATE_MODEL_TASK_NAME
        ):
            information_extraction = InformationExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.create_models()

        if (
            information_extraction_task.task
            == InformationExtraction.SUGGESTIONS_TASK_NAME
        ):
            information_extraction = InformationExtraction(
                information_extraction_task.tenant,
                information_extraction_task.params.property_name,
            )
            return information_extraction.get_suggestions()

        return False, "Error"
