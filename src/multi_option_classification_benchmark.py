import json
import pickle
from os import listdir
from os.path import join
from pathlib import Path
from time import time

import rich
from sklearn.metrics import f1_score

from config import ROOT_PATH, APP_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from metadata_extraction.PdfData import PdfData
from multi_option_extraction.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.filter_segments_methods.CleanBeginningDot500 import CleanBeginningDot500
from multi_option_extraction.multi_labels_methods.SingleLabelBert import SingleLabelBert

from multi_option_extraction.results import get_results_table, add_row

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(Path(__file__).parent, "multi_option_extraction", "labeled_data")
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")


text_extractors = [CleanBeginningDot500]
multi_option_extractors = [SingleLabelBert]
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
    for pdf_name in sorted(get_task_pdf_names()[task_name]):
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
    multi_option_extractions_data: list[MultiOptionData] = get_multi_option_benchmark_data(["d4la_document_type"])

    for multi_option_data in multi_option_extractions_data:
        for method in PDF_TOPIC_CLASSIFICATION_METHODS:
            yield multi_option_data, method


def get_benchmark_custom_methods(repetitions: int = 4):
    results_table = get_results_table()

    for multi_option_extractions_data, method in loop_datasets_methods():
        start = time()
        print("Calculating", multi_option_extractions_data.extraction_identifier, method.get_name())
        performance = method.get_performance(multi_option_extractions_data, repetitions)
        add_row(results_table, method, round(time() - start), performance)


def get_one_hot(multi_option_samples: list[MultiOptionSample], options: list[Option]):
    values = [x.values for x in multi_option_samples]
    return MultiOptionExtractionMethod.one_hot_to_options_list(values, options)


def get_multi_option_extractor_benchmark():
    results_table = get_results_table()

    multi_option_extractions_data: list[MultiOptionData] = get_multi_option_benchmark_data()
    for multi_option_data in multi_option_extractions_data:
        start = time()
        multi_option_extractor = MultiOptionExtractor(extraction_identifier=multi_option_data.extraction_identifier)
        train_set, test_set = MultiOptionExtractionMethod.get_train_test_sets(multi_option_data, 22)
        truth_one_hot = get_one_hot(test_set.samples, multi_option_data.options)

        multi_option_extractor.create_model(train_set)
        test_data = [x.pdf_data for x in test_set.samples]
        multi_option_predictions = multi_option_extractor.get_multi_option_predictions(test_data)
        predictions_one_hot = get_one_hot(multi_option_predictions, multi_option_data.options)

        performance = 100 * f1_score(truth_one_hot, predictions_one_hot, average="micro")

        results_table.add_row("Benchmark", "extractor", f"{round(time() - start / 60, 1)}", f"{round(performance, 2)}%")
        rich.print(results_table)


if __name__ == "__main__":
    get_multi_option_extractor_benchmark()
