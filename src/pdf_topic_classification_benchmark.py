from os.path import join
from config import ROOT_PATH, APP_PATH
from rich import print

from pdf_topic_classification.cache_paragraphs import cache_pdfs_features
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.pdf_topic_classification_methods.NaiveMethod import NaiveMethod
from pdf_topic_classification.results import get_results_table, add_row

CACHE_PARAGRAPHS_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")


def get_results(with_cache_pdf_features: bool = False):
    if with_cache_pdf_features:
        cache_pdfs_features()

    labeled_data = get_labeled_data()
    results_table = get_results_table()

    for labeled_data_task in labeled_data:
        for method in [NaiveMethod]:
            task_name = labeled_data_task.labeled_data_task
            method_instance = method('benchmark', task_name, labeled_data_task.options)
            add_row(results_table, method_instance, method_instance.get_performance(labeled_data_task))

    print(results_table)


if __name__ == "__main__":
    get_results()
