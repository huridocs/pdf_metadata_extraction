import json
import pickle
from os import listdir
from os.path import join
from pathlib import Path
from time import time

from config import ROOT_PATH, APP_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from metadata_extraction.PdfData import PdfData
from multi_option_extraction.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.filter_segments_methods.CleanBeginningDigits3000 import CleanBeginningDigits3000
from multi_option_extraction.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from multi_option_extraction.multi_labels_methods.BertBatch1 import BertBatch1
from multi_option_extraction.multi_labels_methods.TfIdfMethod import TfIdfMethod
from multi_option_extraction.multi_option_extraction_methods.FuzzyLast import FuzzyLast
from multi_option_extraction.multi_option_extraction_methods.FuzzyLastCleanLabel import FuzzyLastCleanLabel

from multi_option_extraction.results import get_results_table, add_row

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(Path(__file__).parent, "multi_option_extraction", "labeled_data")
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")


text_extractors = [CleanBeginningDot1000]
multi_option_extractors = [BertBatch1]
PDF_TOPIC_CLASSIFICATION_METHODS = [
    MultiOptionExtractionMethod(x, y) for x in text_extractors for y in multi_option_extractors
]

# fuzzy_methods = [FirstFuzzyCountry(), All75FuzzyMethod(), All88FuzzyMethod(), All100FuzzyMethod(), FirstFuzzyMethod(), LastFuzzyMethod()]
# fuzzy_methods = [FuzzyFirstCleanLabel()]
# PDF_TOPIC_CLASSIFICATION_METHODS = [FuzzyLast(), FuzzyLastCleanLabel()]


def get_multi_option_benchmark_data(filter_names: list[str] = None) -> list[MultiOptionData]:
    benchmark_data: list[MultiOptionData] = list()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        if filter_names and task_name not in filter_names:
            continue

        print(f"Loading task {task_name}")

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "options.json"), mode="r") as file:
            options = [Option(id=x, label=x) for x in json.load(file)]

        multi_option_samples = get_samples(task_name)
        multi_value: bool = len([sample for sample in multi_option_samples if len(sample.values) > 1]) != 0
        extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name=task_name)
        benchmark_data.append(
            MultiOptionData(
                samples=multi_option_samples,
                options=options,
                multi_value=multi_value,
                extraction_identifier=extraction_identifier,
            )
        )

    return benchmark_data


def get_samples(task_name):
    with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)

    multi_option_samples: list[MultiOptionSample] = list()
    for pdf_name in get_task_pdf_names()[task_name]:
        with open(join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle"), mode="rb") as file:
            pdf_data: PdfData = pickle.load(file)

        values = [Option(id=x, label=x) for x in labels_dict[pdf_name]]
        extraction_sample = MultiOptionSample(pdf_data=pdf_data, values=values)
        multi_option_samples.append(extraction_sample)
    return multi_option_samples


def get_task_pdf_names():
    task_pdf_names: dict[str, set[str]] = dict()

    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            task_pdf_names.setdefault(task_name, set()).update(labels_dict.keys())

    return task_pdf_names


def loop_datasets_methods():
    # cejil_countries
    # cejil_date
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    multi_option_extractions_data: list[MultiOptionData] = get_multi_option_benchmark_data(["cyrilla_keywords", "d4la_document_type"])

    for multi_option_data in multi_option_extractions_data:
        for method in PDF_TOPIC_CLASSIFICATION_METHODS:
            yield multi_option_data, method


def get_benchmark(repetitions: int = 4):
    results_table = get_results_table()

    for multi_option_extractions_data, method in loop_datasets_methods():
        start = time()
        print("Calculating", multi_option_extractions_data.extraction_identifier, method.get_name())
        performance = method.get_performance(multi_option_extractions_data, repetitions)
        add_row(results_table, method, round(time() - start), performance)


if __name__ == "__main__":
    get_benchmark()
