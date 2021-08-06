from typing import List
import numpy as np

from segment_predictor.Segment import Segment


def get_features_one_segment(segment: Segment, page_number: int) -> np.ndarray:
    features_objective_segment = segment.get_features_array()

    return np.concatenate(([page_number],
                           features_objective_segment),
                          axis=None)


def get_training_data(segments: List[Segment]):
    features_array = None
    for segment in segments:
        features = get_features_one_segment(segment, segment.page_number)

        features_with_predictive_variable = np.concatenate((features, np.array([segment.ml_class_label])),
                                                           axis=None)
        features_with_predictive_variable = features_with_predictive_variable.reshape(1, len(
            features_with_predictive_variable))

        if features_array is None:
            features_array = features_with_predictive_variable
        else:
            features_array = np.concatenate((features_array, features_with_predictive_variable), axis=0)

    return features_array


def get_features(segments: List[Segment]):
    training_data = get_training_data(segments)
    if training_data is None:
        return None
    return training_data[:, :-1]
