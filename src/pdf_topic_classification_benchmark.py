from os.path import join
from time import time

from config import ROOT_PATH, APP_PATH
from rich import print

from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.cache_paragraphs import cache_paragraph_extraction_predictions
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.results import get_results_table, add_row

from multi_option_extraction.methods.BertBaseMethod import BertBaseMethod
from multi_option_extraction.methods.FastTextMethod import FastTextMethod
from multi_option_extraction.methods.SetFitMethod import SetFitMethod
from multi_option_extraction.methods.TfIdfMethod import TfIdfMethod
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from pdf_topic_classification.pdf_topic_classification_methods.CheckData import CheckData
from pdf_topic_classification.pdf_topic_classification_methods.NaiveMethod import NaiveMethod
from pdf_topic_classification.text_extraction_methods.SummariesMethod import SummariesMethod
from pdf_topic_classification.text_extraction_methods.TextAtTheBeginningMethod import TextAtTheBeginningMethod
from pdf_topic_classification.text_extraction_methods.TextAtTheEndMethod import TextAtTheEndMethod

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")
PDF_TOPIC_CLASSIFICATION_METHODS = [
    # CheckData(),
    # NaiveMethod(),
    # PdfTopicClassificationMethod(TextAtTheBeginningMethod, BertBaseMethod),
    # PdfTopicClassificationMethod(TextAtTheBeginningMethod, FastTextMethod),
    # PdfTopicClassificationMethod(TextAtTheBeginningMethod, SetFitMethod),
    # PdfTopicClassificationMethod(TextAtTheBeginningMethod, TfIdfMethod),
    # PdfTopicClassificationMethod(TextAtTheEndMethod, BertBaseMethod),
    # PdfTopicClassificationMethod(TextAtTheEndMethod, FastTextMethod),
    # PdfTopicClassificationMethod(TextAtTheEndMethod, SetFitMethod),
    # PdfTopicClassificationMethod(TextAtTheEndMethod, TfIdfMethod),
    # PdfTopicClassificationMethod(SummariesMethod, FastTextMethod),
    # PdfTopicClassificationMethod(SummariesMethod, TfIdfMethod),
    PdfTopicClassificationMethod(SummariesMethod, BertBaseMethod),
    # PdfTopicClassificationMethod(SummariesMethod, SetFitMethod),
]


def get_results(with_cache_paragraph_extraction_predictions: bool = False):
    if with_cache_paragraph_extraction_predictions:
        cache_paragraph_extraction_predictions()

    pdf_topic_classification_labeled_data: list[PdfTopicClassificationLabeledData] = get_labeled_data("cyrilla")
    results_table = get_results_table()

    for labeled_data_one_task in pdf_topic_classification_labeled_data:
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
