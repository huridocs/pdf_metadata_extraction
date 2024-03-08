import random
from os.path import join
from typing import Type

from sklearn.metrics import f1_score

from config import DATA_PATH
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionSample, MultiOptionExtractionData
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor
from multi_option_extraction.PdfMultiOptionExtractionLabeledData import PdfMultiOptionExtractionLabeledData
from multi_option_extraction.TextToMultiOptionMethod import TextToMultiOptionMethod
from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfTextExtractionMethod import PdfTextExtractionMethod


class PdfMultiOptionExtractionMethod:
    def __init__(
        self, text_extraction_method: Type[PdfTextExtractionMethod] = None, multi_option_method: Type[TextToMultiOptionMethod] = None
    ):
        self.multi_option_method = multi_option_method
        self.text_extraction_method = text_extraction_method
        self.run_name = ""
        self.task_name = ""
        self.options = []
        self.multi_value = False
        self.base_path = ""

    def set_parameters(self, run_name: str, pdf_topic_classification_labeled_data: PdfMultiOptionExtractionLabeledData):
        self.run_name = run_name
        self.options = pdf_topic_classification_labeled_data.options
        self.task_name = pdf_topic_classification_labeled_data.task_name
        self.multi_value = pdf_topic_classification_labeled_data.multi_value
        self.base_path = join(DATA_PATH, self.run_name, self.task_name)

    def get_name(self):
        if self.text_extraction_method and self.multi_option_method:
            text_extractor_name = self.text_extraction_method.__name__.replace("Method", "")
            multi_option_name = self.multi_option_method.__name__.replace("Method", "")
            text_extractor_name = text_extractor_name.replace("TextAtThe", "")
            multi_option_name = multi_option_name.replace("TextAtThe", "")
            return f"{text_extractor_name}_{multi_option_name}"

        return self.__class__.__name__

    @staticmethod
    def get_train_test_sets(task_labeled_data, seed: int) -> (list[PdfLabels], list[PdfLabels]):
        train_size = int(len(task_labeled_data.pdfs_labels) * 0.8)
        random.seed(seed)
        train_set: list[PdfLabels] = random.sample(task_labeled_data.pdfs_labels, k=train_size)
        test_set: list[PdfLabels] = [x for x in task_labeled_data.pdfs_labels if x not in train_set]
        return train_set, test_set

    def get_performance(self, task_labeled_data: PdfMultiOptionExtractionLabeledData, repetitions: int = 1) -> float:
        scores = list()
        seeds = [22, 23, 24, 25]
        for i in range(repetitions):
            train_set, test_set = self.get_train_test_sets(task_labeled_data, seeds[i])
            truth_one_hot = self.one_hot_to_options_list([x.labels for x in test_set])

            self.train(train_set)
            predictions = self.predict(test_set)

            predictions_one_hot = self.one_hot_to_options_list(predictions)
            score = f1_score(truth_one_hot, predictions_one_hot, average="micro")
            scores.append(100 * score)
            print(f"Score for {seeds[i]} {self.run_name} {self.task_name} {self.get_name()}: {100 * score}")

        return sum(scores) / len(scores)

    def one_hot_to_options_list(self, pdfs_options: list[list[str]]) -> list[list[int]]:
        options_one_hot: list[list[int]] = list()
        for pdf_options in pdfs_options:
            pdf_options_one_hot = [0] * len(self.options)

            for pdf_option in pdf_options:
                if pdf_option in self.options:
                    pdf_options_one_hot[self.options.index(pdf_option)] = 1

            options_one_hot.append(pdf_options_one_hot)

        return options_one_hot

    def train(self, pdfs_labels: list[PdfLabels]):
        samples: list[MultiOptionExtractionSample] = list()

        for pdf_labels in pdfs_labels:
            text_at_the_beginning = self.text_extraction_method(
                pdf_paragraphs=pdf_labels.paragraphs, options=self.options, multi_option=self.multi_value
            )

            sample = text_at_the_beginning.get_labeled_data_sample(pdf_labels.labels)
            samples.append(sample)

        multi_option_extraction_data = MultiOptionExtractionData(
            multi_value=self.multi_value, options=self.get_options(), samples=samples
        )
        multi_option_extraction = MultiOptionExtractor(self.run_name, self.task_name)
        multi_option_extraction.METHODS = [self.multi_option_method]
        print("Creating model")
        multi_option_extraction.create_model(multi_option_extraction_data)

    def predict(self, pdfs_labels: list[PdfLabels]):
        multi_option_extraction = MultiOptionExtractor(self.run_name, self.task_name)
        multi_option_extraction.METHODS = [self.multi_option_method]
        semantic_prediction_data: list[SemanticPredictionData] = list()
        for pdf_labels in pdfs_labels:
            text_at_the_beginning = self.text_extraction_method(
                pdf_paragraphs=pdf_labels.paragraphs, options=self.options, multi_option=self.multi_value
            )

            sample = text_at_the_beginning.get_semantic_prediction_data()
            semantic_prediction_data.append(sample)

        predictions = multi_option_extraction.get_multi_option_predictions(semantic_prediction_data)

        values: list[list[str]] = [[option.label for option in prediction.values] for prediction in predictions]
        return values

    def get_options(self):
        return [Option(id=option, label=option) for option in self.options]
