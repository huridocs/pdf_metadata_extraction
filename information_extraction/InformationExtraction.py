import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pymongo

from data.LabeledData import LabeledData
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from information_extraction.Segment import Segment
import lightgbm as lgb

from information_extraction.XmlFile import XmlFile
from semantic_information_extraction.SemanticInformationExtraction import SemanticInformationExtraction


class InformationExtraction:
    def __init__(self, tenant: str, extraction_name: str):
        self.tenant = tenant
        self.extraction_name = extraction_name
        self.semantic_information_extraction = SemanticInformationExtraction(tenant, extraction_name)
        self.segments: List[Segment] = list()
        root_folder = f'{Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()}/docker_volume'
        self.model_path = f'{root_folder}/{self.tenant}/{self.extraction_name}/segment_predictor_model/model.model'
        self.model = None
        self.load_model()
        client = pymongo.MongoClient('mongodb://mongo:27017')
        self.pdf_information_extraction_db = client['pdf_information_extraction']

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)

    def set_segments(self):
        client = pymongo.MongoClient('mongodb://mongo:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']

        find_filter = {"extraction_name": self.extraction_name, "tenant": self.tenant}
        for document in pdf_information_extraction_db.labeleddata.find(find_filter, no_cursor_timeout=True):
            segments = XmlFile.get_segments(LabeledData(**document))
            self.segments.extend(segments)

    def create_models(self):
        self.set_segments()
        self.run_light_gbm()

        find_filter = {"extraction_name": self.extraction_name, "tenant": self.tenant}
        semantic_extraction_data: List[SemanticExtractionData] = list()

        for document in self.pdf_information_extraction_db.labeleddata.find(find_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document)
            suggestion = self.get_suggested_segment(labeled_data)
            semantic_extraction_data.append(
                SemanticExtractionData(text=labeled_data.label_text, segment_text=suggestion.segment_text))

        if len(semantic_extraction_data) < 10:
            return

        self.semantic_information_extraction.create_model(semantic_extraction_data)

    @staticmethod
    def get_training_data(segments):
        X = None
        y = np.array([])

        for segment in segments:
            features = segment.get_features_array()
            features = features.reshape(1, len(features))
            X = features if X is None else np.concatenate((X, features), axis=0)
            y = np.append(y, segment.ml_class_label)

        return X, y

    def run_light_gbm(self):
        x_train, y_train = self.get_training_data(self.segments)

        if x_train is None:
            return

        parameters = dict()
        parameters["num_leaves"] = 35
        parameters['feature_fraction'] = 1
        parameters['bagging_fraction'] = 1
        parameters['bagging_freq'] = 5
        parameters["objective"] = 'binary'
        parameters["learning_rate"] = 0.05
        parameters["metric"] = 'binary_logloss'
        parameters['verbose'] = -1
        parameters['boosting_type'] = 'gbdt'

        train_data = lgb.Dataset(x_train, y_train)
        num_round = 3000
        light_gbm_model = lgb.train(parameters, train_data, num_round)

        if not light_gbm_model:
            return

        self.model = light_gbm_model
        if not os.path.exists(Path(self.model_path).parents[0]):
            os.makedirs(Path(self.model_path).parents[0])

        self.model.save_model(self.model_path, num_iteration=self.model.best_iteration)

    def get_suggestions(self) -> List[Suggestion]:
        self.create_models()
        find_filter = {"extraction_name": self.extraction_name, "tenant": self.tenant}
        suggestions: List[Suggestion] = list()

        for document in self.pdf_information_extraction_db.labeleddata.find(find_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document)
            suggestions.append(self.get_suggested_segment(labeled_data))

        for document in self.pdf_information_extraction_db.predictiondata.find(find_filter, no_cursor_timeout=True):
            labeled_data = LabeledData(**document, label_text="",
                                       page_width=0,
                                       page_height=0,
                                       xml_segments_boxes=[],
                                       label_segments_boxes=[])
            suggestions.append(self.get_suggested_segment(labeled_data))

        segments_text = [x.segment_text for x in suggestions]
        texts = self.semantic_information_extraction.get_semantic_predictions(segments_text)
        for index, suggestion in enumerate(suggestions):
            suggestion.text = texts[index]

        shutil.rmtree(XmlFile.get_xml_folder_path(self.tenant, self.extraction_name), ignore_errors=True)
        return suggestions

    def get_suggested_segment(self, labeled_data: LabeledData):
        segments = XmlFile.get_segments(labeled_data)
        if not segments:
            return Suggestion(**labeled_data.dict(), text='', segment_text='')

        x, y = self.get_training_data(segments)
        predictions = self.model.predict(x)
        texts = []
        for index, segment in enumerate(segments):
            if predictions[index] > 0.5:
                texts.append(segment.text_content)
        text = ' '.join(texts)
        return Suggestion(**labeled_data.dict(), text=text, segment_text=text)
