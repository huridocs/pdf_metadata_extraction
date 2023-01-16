import logging
from time import time
from typing import List

import numpy as np

import lightgbm as lgb
from sklearn.metrics import f1_score

from config import config_logger
from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from segment_selector.methods.lightgbm_stack_4.SegmentLightgbmStack4 import SegmentLightgbmStack4


class LightgbmStack4:
    def __init__(self):
        self.segments: List[SegmentLightgbmStack4] = list()
        self.model = None
        self.best_cut = 0

    def create_model(self, training_pdfs_features: List[PdfFeatures]):
        start = time()
        self.set_segments(training_pdfs_features)
        config_logger.info(f"Set segments {int(time() - start)} seconds")

        if len(self.segments) == 0:
            return None

        start = time()
        self.run_light_gbm()
        config_logger.info(f"Run lightGBM {int(time() - start)} seconds")

        return self.model

    @staticmethod
    def set_model(x_train, y_train):
        parameters = dict()
        parameters["num_leaves"] = 70
        parameters["feature_fraction"] = 0.9
        parameters["bagging_fraction"] = 0.9
        parameters["bagging_freq"] = 0
        parameters["objective"] = "binary"
        parameters["learning_rate"] = 0.05
        parameters["metric"] = "binary_logloss"
        parameters["verbose"] = -1
        parameters["boosting_type"] = "gbdt"

        train_data = lgb.Dataset(x_train, y_train)
        num_round = 3000
        light_gbm_model = lgb.train(parameters, train_data, num_round)

        return light_gbm_model

    def run_light_gbm(self):
        x_train, y_train = self.get_training_data()
        self.model = self.set_model(x_train, y_train)

    def get_training_data(self):
        y = np.array([])
        x_rows = []
        for segment in self.segments:
            x_rows.append(segment.get_features_array())
            y = np.append(y, segment.pdf_segment.ml_label)

        X = np.zeros((len(x_rows), len(x_rows[0]) if x_rows else 0))
        for i, v in enumerate(x_rows):
            X[i] = v

        return X, y

    def set_segments(self, pdfs_features: List[PdfFeatures]):
        self.segments = list()
        for pdf_features in pdfs_features:
            self.segments.extend(SegmentLightgbmStack4.from_pdf_features(pdf_features))

    def predict(self, model, testing_pdfs_features: List[PdfFeatures]):
        self.set_segments(testing_pdfs_features)
        x, y = self.get_training_data()
        x = x[:, : model.num_feature()]
        predictions = model.predict(x)
        return predictions

    @staticmethod
    def get_performance(predictions, y_truth):
        return f1_score(y_truth, [prediction > 0.5 for prediction in predictions])
