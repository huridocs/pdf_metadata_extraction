from os.path import dirname, realpath, join
from typing import List, Type

import pandas as pd
from datetime import datetime

from data.SemanticExtractionData import SemanticExtractionData
from performance.Results import Results
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.SameInputOutputMethod import SameInputOutputMethod
from semantic_metadata_extraction.methods.MT5EnglishSpanishMethod import MT5EnglishSpanishMethod
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod
from semantic_metadata_extraction.methods.T5Method import T5Method
from semantic_metadata_extraction.methods.T5TransformersLowercaseMethod import T5TransformersLowercaseMethod
from semantic_metadata_extraction.methods.T5TransformersMethod import T5TransformersMethod
from semantic_metadata_extraction.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from semantic_metadata_extraction.methods.T5ZeroShot import T5ZeroShot


class CheckPerformance:
    SCRIPT_PATH = dirname(realpath(__file__))

    TENANT = "check_performance"

    METHODS: List[Type[Method]] = [
        # SameInputOutputMethod,
        # RegexMethod,
        # DateParserMethod,
        # T5Method,
        # T5TransformersMethod,
        # T5TransformersLowercaseMethod,
        # MT5EnglishSpanishMethod,
        # MT5TrueCaseEnglishSpanishMethod,
        T5ZeroShot,
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

    TRAINING_SET_LENGTH = 30

    def __init__(self):
        self.semantic_information_data = None
        self.current_dataset = None
        self.current_method = None
        self.current_prediction = None
        self.current_prediction = None

        prefix = f"{datetime.now():%Y_%m_%d_%H_%M}"

        self.all_results = Results(results_name=f"all_results____{prefix}", training_set_length=self.TRAINING_SET_LENGTH)
        self.best_results = Results(results_name=f"best_results____{prefix}", training_set_length=self.TRAINING_SET_LENGTH)

    def dataset_name_to_property_name(self):
        return self.current_dataset.replace("_", "").replace(".tsv", "")

    def get_semantic_extraction_data(self, file_name):
        df = pd.read_csv(join(self.SCRIPT_PATH, "performance", "datasets", file_name), sep="\t")
        self.semantic_information_data = []
        for index, row in df.iterrows():
            semantic_extraction_data = SemanticExtractionData(
                text=row["output"], segment_text=row["input"], language_iso=row["language"]
            )
            self.semantic_information_data.append(semantic_extraction_data)

    def run(self):
        for self.current_dataset in self.DATASETS:
            self.get_semantic_extraction_data(self.current_dataset)
            self.best_results.set_start_time()

            names = [list()]
            accuracies = list()

            for self.current_method in self.METHODS:
                self.all_results.set_start_time()
                method_instance = self.current_method(self.TENANT, self.dataset_name_to_property_name())
                accuracy, self.current_prediction = method_instance.performance(self.semantic_information_data, self.TRAINING_SET_LENGTH)
                accuracies.append(accuracy)
                names.append(method_instance.get_name())
                self.all_results.save_result(self.current_dataset, names[-1], accuracies[-1])
                self.write_mistakes()

            self.best_results.save_result(self.current_dataset, names[accuracies.index(max(accuracies))], max(accuracies))

        self.write_results()

    def write_results(self):
        self.all_results.write_results()
        self.best_results.write_results()

    def write_mistakes(self):
        performance_train_set, performance_test_set = Method.get_train_test(self.semantic_information_data,
                                                                            self.TRAINING_SET_LENGTH)
        correct_path = f"../performance_results/mistakes/{self.current_method}_{self.current_dataset}_correct.txt"
        mistakes_path = f"../performance_results/mistakes/{self.current_method}_{self.current_dataset}_mistakes.txt"

        with open(mistakes_path, "w") as input_file:
            input_file.write("Mistakes\n\n")

        with open(correct_path, "w") as input_file:
            input_file.write("Correct\n\n")

        for prediction, semantic_information_data in zip(self.current_prediction, performance_test_set):
            text_one_line = semantic_information_data.segment_text.replace("\n", " ")

            file_name = correct_path if prediction == semantic_information_data.text else mistakes_path

            with open(file_name, "a") as input_file:
                line = f"prediction:{prediction}\ntarget:{semantic_information_data.text}\ntext:{text_one_line}\n\n"
                input_file.write(line)


if __name__ == "__main__":
    check_performance = CheckPerformance()
    check_performance.run()
