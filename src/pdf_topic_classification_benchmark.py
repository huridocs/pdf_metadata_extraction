from os.path import join
from time import time


from config import ROOT_PATH, APP_PATH

from multi_option_extraction.methods.BertSeqSteps import BertSeqSteps
from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from pdf_topic_classification.cache_pdf_features import cache_paragraph_extraction_predictions
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data

from pdf_topic_classification.results import get_results_table, add_row
from pdf_topic_classification.text_extraction_methods.CleanBeginningDot750 import CleanBeginningDot750

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

text_extractors = [CleanBeginningDot750]
multi_option_extractors = [BertSeqSteps]

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


def get_benchmark(repetitions: int = 4, with_cache_paragraph_extraction_predictions: bool = False):
    if with_cache_paragraph_extraction_predictions:
        cache_paragraph_extraction_predictions()

    results_table = get_results_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)
        start = time()
        print("Calculating", method.task_name, method.get_name())
        performance = method.get_performance(labeled_data_one_task, repetitions)
        add_row(results_table, method, round(time() - start), performance)


if __name__ == "__main__":
    get_benchmark()

