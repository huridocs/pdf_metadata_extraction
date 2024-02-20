from os.path import join
from time import time

from config import ROOT_PATH, APP_PATH
from rich import print

from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.cache_paragraphs import cache_paragraph_extraction_predictions
from pdf_topic_classification.methods import PDF_TOPIC_CLASSIFICATION_METHODS
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.results import get_results_table, add_row

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")


def get_results(with_cache_paragraph_extraction_predictions: bool = False):
    if with_cache_paragraph_extraction_predictions:
        cache_paragraph_extraction_predictions()

    pdf_topic_classification_labeled_data: list[PdfTopicClassificationLabeledData] = get_labeled_data()
    results_table = get_results_table()

    for labeled_data_one_task in pdf_topic_classification_labeled_data[:1]:
        for method in PDF_TOPIC_CLASSIFICATION_METHODS:
            method.set_parameters("benchmark", labeled_data_one_task)
            start = time()
            print("Calculating", method.task_name, method.get_name())
            performance = method.get_performance(labeled_data_one_task)
            add_row(results_table, method, round(time() - start), performance)
            print("finished in", round(time() - start, 1), "seconds")

        add_row(results_table)

    print(results_table)


if __name__ == "__main__":
    get_results()
