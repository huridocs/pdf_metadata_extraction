from os.path import join
from time import time

import rich

from config import ROOT_PATH, APP_PATH

from multi_option_extraction.methods.BertBatch1FitTokens import BertBatch1FitTokens
from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.cache_pdf_features import cache_paragraph_extraction_predictions
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.pdf_topic_classification_methods.FuzzySegmentSelectorMethod import FuzzySegmentSelectorMethod
from pdf_topic_classification.results import get_results_table, add_row, get_predictions_table, add_prediction_row

from pdf_topic_classification.text_extraction_methods.CleanBeginningDot500 import CleanBeginningDot500

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

text_extractors = [CleanBeginningDot500]
multi_option_extractors = [BertBatch1FitTokens]

PDF_TOPIC_CLASSIFICATION_METHODS = [FuzzySegmentSelectorMethod()]
# PDF_TOPIC_CLASSIFICATION_METHODS = [All100FuzzyMethod(), All75FuzzyMethod()] + [PdfTopicClassificationMethod(x, y) for x in text_extractors for y in multi_option_extractors]


def loop_datasets_methods():
    pdf_topic_classification_labeled_data: list[PdfTopicClassificationLabeledData] = get_labeled_data("cejil_president")

    for labeled_data_one_task in pdf_topic_classification_labeled_data:
        for method in PDF_TOPIC_CLASSIFICATION_METHODS:
            yield labeled_data_one_task, method


def get_results(with_cache_paragraph_extraction_predictions: bool = False):
    if with_cache_paragraph_extraction_predictions:
        cache_paragraph_extraction_predictions()

    results_table = get_results_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)
        start = time()
        print("Calculating", method.task_name, method.get_name())
        performance = method.get_performance(labeled_data_one_task, 4)
        add_row(results_table, method, round(time() - start), performance)


def check_mistakes():
    predictions_table = get_predictions_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)

        print("Calculating", method.task_name, method.get_name())

        train, test_set = method.get_train_test_sets(labeled_data_one_task, 22)
        predictions = method.predict(test_set)
        labels = [x.labels for x in test_set]
        pdfs_names = [x.pdf_name for x in labeled_data_one_task.pdfs_labels]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(predictions_table, pdf_name, label, prediction)

        add_prediction_row(predictions_table)

    rich.print(predictions_table)


if __name__ == "__main__":
    get_results()
