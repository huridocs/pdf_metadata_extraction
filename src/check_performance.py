from os.path import dirname, realpath, join
from typing import List, Type

import pandas as pd
from datetime import datetime

from data.SemanticExtractionData import SemanticExtractionData
from performance.Results import Results
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.SameInputOutputMethod import SameInputOutputMethod
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod
from semantic_metadata_extraction.methods.DistilBertSpanishMethod import DistilBertSpanishMethod
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod
from semantic_metadata_extraction.methods.T5Method import T5Method
from semantic_metadata_extraction.methods.T5Method5Epochs import T5Method5Epochs

SCRIPT_PATH = dirname(realpath(__file__))

TENANT = "check_performance"
METHODS: List[Type[Method]] = [
    SameInputOutputMethod,
    RegexMethod,
    DateParserMethod,
    DistilBertSpanishMethod,
    T5Method,
    T5Method5Epochs,
]
DATASETS: List[str] = [
    "code_spanish.tsv",
    "country_spanish.tsv",
    "date_spanish.tsv",
    "document_code.tsv",
    "judge_name.tsv",
    "vote_english.tsv",
    "year_multilingual.tsv",
]

RESULTS_PREFIX = f"{datetime.now():%Y_%m_%d_%H_%M}"


def get_semantic_extraction_data(file_name):
    df = pd.read_csv(join(SCRIPT_PATH, "performance", "datasets", file_name), sep="\t")
    semantic_extraction_data_list = []
    for index, row in df.iterrows():
        semantic_extraction_data = SemanticExtractionData(
            text=row["output"], segment_text=row["input"], language_iso=row["language"]
        )
        semantic_extraction_data_list.append(semantic_extraction_data)

    return semantic_extraction_data_list


def dataset_name_to_property_name(dataset_name: str):
    return dataset_name.replace("_", "").replace(".tsv", "")


def check_performance():
    training_set_length = 30
    all_results = Results(results_name="all_results____" + RESULTS_PREFIX, training_set_length=training_set_length)
    best_results = Results(results_name="best_results____" + RESULTS_PREFIX, training_set_length=training_set_length)

    for dataset in DATASETS:
        semantic_information_data_list = get_semantic_extraction_data(dataset)
        best_results.set_start_time()

        names = []
        accuracies = []

        print("Performance on", dataset)
        for method in METHODS:
            all_results.set_start_time()
            method_instance = method(TENANT, dataset_name_to_property_name(dataset))
            accuracies.append(method_instance.performance(semantic_information_data_list, training_set_length))
            names.append(method_instance.get_name())
            all_results.save_result(dataset, names[-1], accuracies[-1])

        best_results.save_result(dataset, names[accuracies.index(max(accuracies))], max(accuracies))

    all_results.write_results()
    best_results.write_results()


if __name__ == "__main__":
    check_performance()
