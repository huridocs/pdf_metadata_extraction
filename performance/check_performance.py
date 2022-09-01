from os import listdir
from os.path import dirname, realpath, join
from typing import List, Type

import pandas as pd

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.RegexMethod import RegexMethod

TENANT = "check_performance"
PROPERTY_NAME = "check_performance"
DATASETS = []
METHODS: List[Type[Method]] = [RegexMethod]
SCRIPT_PATH = dirname(realpath(__file__))


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
    for dataset in listdir(join(SCRIPT_PATH, "datasets")):
        semantic_information_data_list = get_semantic_extraction_data(dataset)
        performances = []
        methods_names = []
        for method in METHODS:
            method_instance = method(TENANT, dataset_name_to_property_name(dataset))
            methods_names.append(method_instance.get_name())
            performances.append(method_instance.performance(semantic_information_data_list))
        print(dataset, max(performances), methods_names[performances.index(max(performances))])


if __name__ == "__main__":
    check_performance()
