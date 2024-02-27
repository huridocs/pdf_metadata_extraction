from os.path import join
from time import time

from config import ROOT_PATH, APP_PATH
from rich import print

from multi_option_extraction.methods.MultiBert import MultiBert
from multi_option_extraction.methods.MultiBertBatch1 import MultiBertBatch1
from multi_option_extraction.methods.MultiBertBatch2 import MultiBertBatch2
from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.cache_paragraphs import cache_paragraph_extraction_predictions
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.results import get_results_table, add_row, get_predictions_table, add_prediction_row

from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from pdf_topic_classification.text_extraction_methods.Beginning750 import Beginning750
from pdf_topic_classification.text_extraction_methods.CleanBeginningDot1500 import CleanBeginningDot1500
from pdf_topic_classification.text_extraction_methods.CleanBeginningDot500 import CleanBeginningDot500
from pdf_topic_classification.text_extraction_methods.CleanBeginningDot750 import CleanBeginningDot750

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")


PDF_TOPIC_CLASSIFICATION_METHODS = [
    PdfTopicClassificationMethod(CleanBeginningDot500, MultiBertBatch2),
    PdfTopicClassificationMethod(CleanBeginningDot500, MultiBertBatch1),
    PdfTopicClassificationMethod(CleanBeginningDot1500, MultiBert),

    PdfTopicClassificationMethod(CleanBeginningDot750, MultiBertBatch1),
    PdfTopicClassificationMethod(CleanBeginningDot750, MultiBertBatch2),
    PdfTopicClassificationMethod(CleanBeginningDot1500, MultiBert),

    PdfTopicClassificationMethod(CleanBeginningDot750, MultiBertBatch2),
    PdfTopicClassificationMethod(CleanBeginningDot1500, MultiBertBatch2),
    PdfTopicClassificationMethod(CleanBeginningDot1500, MultiBert),
    PdfTopicClassificationMethod(Beginning750, MultiBert),
    PdfTopicClassificationMethod(Beginning750, MultiBertBatch2),
    PdfTopicClassificationMethod(Beginning750, MultiBertBatch1),
]

def loop_datasets_methods():
    pdf_topic_classification_labeled_data: list[PdfTopicClassificationLabeledData] = get_labeled_data("cyrilla")

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
        performance = method.get_performance(labeled_data_one_task)
        add_row(results_table, method, round(time() - start), performance)
        print("finished in", round(time() - start, 1), "seconds")

    print(results_table)


def check_mistakes():
    predictions_table = get_predictions_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)

        print("Calculating", method.task_name, method.get_name())
        train, test_set = method.get_train_test_sets(labeled_data_one_task)
        predictions = method.predict(test_set)
        labels = [x.labels for x in test_set]
        pdfs_names = [x.pdf_name for x in labeled_data_one_task.pdfs_labels]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(predictions_table, pdf_name, label, prediction)

        add_row(predictions_table)

    print(predictions_table)


if __name__ == "__main__":
    get_results()
