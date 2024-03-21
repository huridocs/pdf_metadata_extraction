import random
from typing import Type

from sklearn.metrics import f1_score

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.MultiLabelMethod import MultiLabelMethod
from multi_option_extraction.FilterSegmentsMethod import FilterSegmentsMethod


class MultiOptionExtractionMethod:
    def __init__(
        self,
        filter_segments_method: Type[FilterSegmentsMethod] = None,
        multi_label_method: Type[MultiLabelMethod] = None,
    ):
        self.multi_label_method = multi_label_method
        self.filter_segments_method = filter_segments_method
        self.extraction_identifier = ExtractionIdentifier(run_name="not set", extraction_name="not set")
        self.options: list[Option] = []
        self.multi_value = False
        self.base_path = ""

    def set_parameters(self, multi_option_data: MultiOptionData):
        self.extraction_identifier = multi_option_data.extraction_identifier
        self.options = multi_option_data.options
        self.multi_value = multi_option_data.multi_value
        self.base_path = multi_option_data.extraction_identifier.get_path()

    def get_name(self):
        if self.filter_segments_method and self.multi_label_method:
            text_extractor_name = self.filter_segments_method.__name__.replace("Method", "")
            multi_option_name = self.multi_label_method.__name__.replace("Method", "")
            text_extractor_name = text_extractor_name.replace("TextAtThe", "")
            multi_option_name = multi_option_name.replace("TextAtThe", "")
            return f"{text_extractor_name}_{multi_option_name}"

        return self.__class__.__name__

    @staticmethod
    def get_train_test_sets(multi_option_data: MultiOptionData, seed: int) -> (MultiOptionData, MultiOptionData):
        if len(multi_option_data.samples) < 15:
            return multi_option_data, multi_option_data

        train_size = int(len(multi_option_data.samples) * 0.8)
        random.seed(seed)

        train_set: list[MultiOptionSample] = random.sample(multi_option_data.samples, k=train_size)[:80]
        test_set: list[MultiOptionSample] = [x for x in multi_option_data.samples if x not in train_set][:30]

        train_data = MultiOptionData(
            samples=train_set,
            options=multi_option_data.options,
            multi_value=multi_option_data.multi_value,
            extraction_identifier=multi_option_data.extraction_identifier,
        )

        test_data = MultiOptionData(
            samples=test_set,
            options=multi_option_data.options,
            multi_value=multi_option_data.multi_value,
            extraction_identifier=multi_option_data.extraction_identifier,
        )

        return train_data, test_data

    def get_performance(self, multi_option_data: MultiOptionData, repetitions: int = 1) -> float:
        self.set_parameters(multi_option_data)
        scores = list()
        seeds = [22, 23, 24, 25]
        for i in range(repetitions):
            train_set, test_set = self.get_train_test_sets(multi_option_data, seeds[i])
            truth_one_hot = self.one_hot_to_options_list([x.values for x in test_set.samples])

            self.train(train_set)
            predictions = self.predict(test_set)

            if not self.multi_value:
                predictions = [x[:1] for x in predictions]

            predictions_one_hot = self.one_hot_to_options_list(predictions)
            score = f1_score(truth_one_hot, predictions_one_hot, average="micro")
            scores.append(100 * score)
            print(f"Score for seed={seeds[i]} {self.extraction_identifier} {self.get_name()}: {100 * score}")

        return sum(scores) / len(scores)

    def one_hot_to_options_list(self, pdfs_options: list[list[Option]]) -> list[list[int]]:
        options_one_hot: list[list[int]] = list()
        option_labels = [x.label for x in self.options]
        for pdf_options in pdfs_options:
            pdf_options_one_hot = [0] * len(self.options)

            for pdf_option in pdf_options:
                if pdf_option.label in option_labels:
                    pdf_options_one_hot[option_labels.index(pdf_option.label)] = 1

            options_one_hot.append(pdf_options_one_hot)

        return options_one_hot

    def train(self, multi_option_data: MultiOptionData):
        self.set_parameters(multi_option_data)

        print("Filtering segments")
        filtered_multi_option_data = self.filter_segments_method().filter(multi_option_data)

        print("Creating model")
        multi_label = self.multi_label_method(self.extraction_identifier, self.options, self.multi_value)
        multi_label.train(filtered_multi_option_data)

    def predict(self, multi_option_data: MultiOptionData) -> list[list[Option]]:
        self.set_parameters(multi_option_data)

        print("Filtering segments")
        filtered_multi_option_data = self.filter_segments_method().filter(multi_option_data)

        print("Prediction")
        multi_label = self.multi_label_method(self.extraction_identifier, self.options, self.multi_value)
        predictions = multi_label.predict(filtered_multi_option_data)

        return predictions
