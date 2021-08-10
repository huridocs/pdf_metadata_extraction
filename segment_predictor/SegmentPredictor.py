import os
from pathlib import Path
from typing import List

import numpy as np
from lightgbm.basic import LightGBMError

from segment_predictor.Segment import Segment
import lightgbm as lgb


class SegmentPredictor:
    def __init__(self, tenant: str, extraction_name: str):
        self.extraction_name = extraction_name
        self.tenant = tenant
        self.segments: List[Segment] = list()
        root_folder = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()
        self.model_path = f'{root_folder}/docker_volume/{self.tenant}/segment_predictor_model/model.model'
        self.xml_paths = f'{root_folder}/docker_volume/{self.tenant}/xml_files/'
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = lgb.Booster(model_file=self.model_path)
        except LightGBMError:
            pass

    def set_segments(self):
        pass

    def create_model(self):
        self.set_segments()
        self.run_light_gbm()

    def get_training_data(self):
        features_array = None
        for segment in self.segments:
            features = self.get_features_one_segment(segment, segment.page_number)

            features_with_predictive_variable = np.concatenate((features, np.array([segment.ml_class_label])),
                                                               axis=None)
            features_with_predictive_variable = features_with_predictive_variable.reshape(1, len(
                features_with_predictive_variable))

            if features_array is None:
                features_array = features_with_predictive_variable
            else:
                features_array = np.concatenate((features_array, features_with_predictive_variable), axis=0)

        return features_array

    @staticmethod
    def get_features_one_segment(segment: Segment, page_number: int) -> np.ndarray:
        features_objective_segment = segment.get_features_array()

        return np.concatenate(([page_number],
                               features_objective_segment),
                              axis=None)

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
        bst = lgb.train(parameters, train_data, num_round)

        if not bst:
            return

        self.model = bst
        self.model.save_model(self.model_path, num_iteration=bst.best_iteration)

