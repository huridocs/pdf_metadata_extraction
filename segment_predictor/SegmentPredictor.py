import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pymongo
from lightgbm.basic import LightGBMError

from data.LabeledData import LabeledData
from segment_predictor.Segment import Segment
import lightgbm as lgb

from segment_predictor.XmlFile import XmlFile


class SegmentPredictor:
    def __init__(self, tenant: str, extraction_name: str):
        self.extraction_name = extraction_name
        self.tenant = tenant
        self.segments: List[Segment] = list()
        root_folder = f'{Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()}/docker_volume'
        self.model_path = f'{root_folder}/{self.tenant}/{self.extraction_name}/segment_predictor_model/model.model'
        self.model = None
        self.load_model()

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

    def create_model(self):
        self.set_segments()
        self.run_light_gbm()
        XmlFile(tenant=self.tenant, extraction_name=self.extraction_name).remove_files()

    def get_training_data(self):
        X = None
        y = np.array([])

        for segment in self.segments:
            features = segment.get_features_array()
            features = features.reshape(1, len(features))
            X = features if X is None else np.concatenate((X, features), axis=0)
            y = np.append(y, segment.ml_class_label)

        return X, y

    def run_light_gbm(self):
        x_train, y_train = self.get_training_data()

        if x_train is None:
            return

        parameters = dict()
        parameters["num_leaves"] = 35
        parameters['feature_fraction'] = 1
        parameters['bagging_fraction'] = 1
        parameters['bagging_freq'] = 5
        parameters["objective"] = 'multiclass'
        parameters["learning_rate"] = 0.05
        parameters["metric"] = {'multi_logloss'}
        parameters['num_class'] = 4
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
