from os import listdir
from os.path import dirname, realpath, join
from typing import List, Type

import pandas as pd
from datetime import datetime

from data.SemanticExtractionData import SemanticExtractionData
from performance.Results import Results
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.methods.T5MethodEarlyStopping import T5MethodEarlyStopping

SCRIPT_PATH = dirname(realpath(__file__))

TENANT = "check_performance"
# METHODS: List[Type[Method]] = [SameInputOutputMethod, RegexMethod, DateParserMethod, DistilBertSpanishMethod, T5Method, T5Method5Epochs]
METHODS: List[Type[Method]] = [T5MethodEarlyStopping]
RESULTS_PREFIX = f"{datetime.now():%Y_%m_%d_%H_%M}"


def get_semantic_extraction_data(file_name):
    df = pd.read_csv(join(SCRIPT_PATH, "datasets", file_name), sep="\t")
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

    for dataset in listdir(join(SCRIPT_PATH, "datasets")):
        semantic_information_data_list = get_semantic_extraction_data(dataset)
        best_results.set_start_time()

        names = []
        accuracies = []

        print(dataset)
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
