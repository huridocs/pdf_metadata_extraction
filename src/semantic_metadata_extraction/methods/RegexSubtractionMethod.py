import re


from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.Method import Method

from tdda import *


class RegexSubtractionMethod(Method):
    def performance(self, semantic_extraction_data: list[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0, []

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        self.train(performance_train_set)
        predictions = self.predict([x.to_semantic_prediction() for x in performance_test_set])
        self.log_performance_sample(semantic_extractions_data=performance_test_set, predictions=predictions)
        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: list[SemanticExtractionData]):
        front_subtraction = [
            self.get_first_subtraction_characters(self.get_text_from_pdf_tags(x.pdf_tags), x.text)
            for x in semantic_extraction_data
        ]
        front_regex_list = rexpy.extract([x for x in front_subtraction if x])
        front_regex_list = [regex[:-1] for regex in front_regex_list]

        back_subtraction = [
            self.get_last_subtraction_characters(self.get_text_from_pdf_tags(x.pdf_tags), x.text)
            for x in semantic_extraction_data
        ]
        back_regex_list = rexpy.extract([x for x in back_subtraction if x])
        back_regex_list = [regex[1:] for regex in back_regex_list]

        self.save_json("regex_subtraction_list.json", front_regex_list + back_regex_list)

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[str]:
        regex_list = self.load_json("regex_subtraction_list.json")
        predictions = [self.get_text_from_pdf_tags(x.pdf_tags) for x in semantic_predictions_data]
        for i, prediction in enumerate(predictions):
            for regex in regex_list:
                matches = re.search(regex, prediction)
                if matches and not matches.start():
                    prediction = prediction[matches.end() :]
                    continue
                if matches and matches.end() == len(prediction):
                    prediction = prediction[: matches.start()]

            predictions[i] = prediction

        return predictions

    @staticmethod
    def get_first_subtraction_characters(segment_text: str, text: str):
        if text not in segment_text:
            return ""

        if text == segment_text:
            return ""

        first_index = segment_text.find(text)

        if not first_index:
            return ""

        return segment_text[:first_index]

    @staticmethod
    def get_last_subtraction_characters(segment_text: str, text: str):
        if text not in segment_text:
            return ""

        if text == segment_text:
            return ""

        first_index = segment_text.find(text) + len(text)

        if not first_index:
            return ""

        return segment_text[first_index:]
