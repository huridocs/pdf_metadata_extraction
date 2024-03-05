from os.path import join
from time import time

import rich

from config import ROOT_PATH, APP_PATH
from multi_option_extraction.methods.DebertaDeepSpeed import DebertaDeepSpeed
from multi_option_extraction.methods.DebertaSequence import DebertaSequence
from multi_option_extraction.methods.DebertaSequenceEarlyStopping import DebertaSequenceEarlyStopping
from multi_option_extraction.methods.SingleLabelBertBatch1 import SingleLabelBertBatch1
from multi_option_extraction.methods.SingleLabelDebertaBatch1 import SingleLabelDebertaBatch1
from multi_option_extraction.methods.SingleLabelDistilbertBatch1 import SingleLabelDistilbertBatch1
from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from pdf_topic_classification.cache_pdf_features import cache_paragraph_extraction_predictions
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data

from pdf_topic_classification.results import get_results_table, add_row, get_predictions_table, add_prediction_row
from pdf_topic_classification.text_extraction_methods.CleanBeginningDot1500 import CleanBeginningDot1500

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

text_extractors = [CleanBeginningDot1500]
multi_option_extractors = [DebertaDeepSpeed]

# fuzzy_methods = [FirstFuzzyCountry(), All75FuzzyMethod(), All88FuzzyMethod(), All100FuzzyMethod(), FirstFuzzyMethod(), LastFuzzyMethod()]
# fuzzy_methods = [FuzzyFirstCleanLabel()]
# PDF_TOPIC_CLASSIFICATION_METHODS = fuzzy_methods


PDF_TOPIC_CLASSIFICATION_METHODS = [
    PdfTopicClassificationMethod(x, y) for x in text_extractors for y in multi_option_extractors
]


def loop_datasets_methods():
    # cejil_countries
    # cejil_date
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    pdf_topic_classification_labeled_data: list[PdfTopicClassificationLabeledData] = get_labeled_data(["cyrilla_keywords"])

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
        performance = method.get_performance(labeled_data_one_task, 1)
        add_row(results_table, method, round(time() - start), performance)


def check_mistakes():
    predictions_table = get_predictions_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)

        print("Calculating", method.task_name, method.get_name())

        train, test_set = method.get_train_test_sets(labeled_data_one_task, 25)
        predictions = method.predict(test_set)
        labels = [x.labels for x in test_set]
        pdfs_names = [x.pdf_name for x in test_set]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(predictions_table, pdf_name, label, prediction)

        add_prediction_row(predictions_table)

    rich.print(predictions_table)


if __name__ == "__main__":
    get_results()
