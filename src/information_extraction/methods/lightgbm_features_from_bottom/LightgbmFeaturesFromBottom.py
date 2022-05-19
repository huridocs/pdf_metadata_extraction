from typing import List

import numpy as np

import lightgbm as lgb

from information_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from information_extraction.methods.lightgbm_features_from_bottom.SegmentLightgbmFeaturesFromBottom import \
    SegmentLightgbmFeaturesFromBottom


class LightgbmFeaturesFromBottom:
    def __init__(self):
        self.segments: List[SegmentLightgbmFeaturesFromBottom] = list()

    def create_model(self, training_pdfs_features: List[PdfFeatures]):
        self.set_segments(training_pdfs_features)
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

        return light_gbm_model

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
            self.segments.extend(SegmentLightgbmFeaturesFromBottom.from_pdf_features(pdf_features))

    def predict(self, model, testing_pdfs_features: List[PdfFeatures]):
        self.set_segments(testing_pdfs_features)
        x, y = self.get_training_data()
        predictions = model.predict(x)
        return predictions
