from os.path import join
from time import time


from config import ROOT_PATH, APP_PATH
from multi_option_extraction.PdfMultiOptionExtractionLabeledData import PdfMultiOptionExtractionLabeledData
from multi_option_extraction.PdfMultiOptionExtractionMethod import PdfMultiOptionExtractionMethod
from multi_option_extraction.get_test_labeled_data import get_labeled_data
from multi_option_extraction.pdf_text_extraction_methods.CleanBeginningDot750 import CleanBeginningDot750
from multi_option_extraction.text_to_multi_option_methods.BertBatch1 import BertBatch1
from multi_option_extraction.text_to_multi_option_methods.BertSeqLikeBert1 import BertSeqLikeBert1
from multi_option_extraction.text_to_multi_option_methods.BertSeqLikeBert1SameLength import BertSeqLikeBert1SameLength

from multi_option_extraction.cache_pdf_features import cache_paragraph_extraction_predictions

from multi_option_extraction.results import get_results_table, add_row

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

text_extractors = [CleanBeginningDot750]
multi_option_extractors = [BertSeqLikeBert1SameLength, BertSeqLikeBert1, BertBatch1]

# fuzzy_methods = [FirstFuzzyCountry(), All75FuzzyMethod(), All88FuzzyMethod(), All100FuzzyMethod(), FirstFuzzyMethod(), LastFuzzyMethod()]
# fuzzy_methods = [FuzzyFirstCleanLabel()]
# PDF_TOPIC_CLASSIFICATION_METHODS = fuzzy_methods


PDF_TOPIC_CLASSIFICATION_METHODS = [
    PdfMultiOptionExtractionMethod(x, y) for x in text_extractors for y in multi_option_extractors
]


def loop_datasets_methods():
    # cejil_countries
    # cejil_date
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    pdf_topic_classification_labeled_data: list[PdfMultiOptionExtractionLabeledData] = get_labeled_data(["cyrilla_keywords"])

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

