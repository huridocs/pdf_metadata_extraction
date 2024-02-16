import random
from abc import ABC, abstractmethod

from sklearn.metrics import f1_score

from pdf_topic_classification.PdfLabels import PdfLabels
from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData


class PdfTopicClassificationMethod(ABC):

    def __init__(self, run_name: str, task_name: str, options: list[str]):
        self.options = options
        self.run_name = run_name
        self.task_name = task_name

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def train(self, pdfs_labels: list[PdfLabels]):
        pass

    @abstractmethod
    def predict(self, pdfs_labels: list[PdfLabels]):
        pass

    def get_performance(self, task_labeled_data: PdfTopicClassificationLabeledData) -> float:
        train_size = int(len(task_labeled_data.pdfs_labels) * 0.8)
        random.seed(22)
        train_set: list[PdfLabels] = random.choices(task_labeled_data.pdfs_labels, k=train_size)
        test_set: list[PdfLabels] = [x for x in task_labeled_data.pdfs_labels if x not in train_set]
        truth_one_hot = self.one_hot_to_options_list([x.labels for x in test_set])
        self.train(train_set)
        predictions = self.predict(test_set)
        predictions_one_hot = self.one_hot_to_options_list(predictions)
        return 100 * f1_score(truth_one_hot, predictions_one_hot, average="macro")

    def one_hot_to_options_list(self, pdfs_options: list[list[str]]) -> list[list[int]]:
        options_one_hot: list[list[int]] = list()
        for pdf_options in pdfs_options:
            pdf_options_one_hot = [0] * len(self.options)

            for pdf_option in pdf_options:
                if pdf_option in self.options:
                    pdf_options_one_hot[self.options.index(pdf_option)] = 1

            options_one_hot.append(pdf_options_one_hot)

        return options_one_hot

