import re


from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from extractors.text_to_text_extractor.TextToTextMethod import TextToTextMethod

from tdda import *


class RegexMethod(TextToTextMethod):
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
        regex_list = rexpy.extract([x.text for x in semantic_extraction_data])
        regex_list = [regex[1:-1] for regex in regex_list]
        self.save_json("regex_list.json", regex_list)

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[str]:
        predictions = ["" for _ in semantic_predictions_data]
        regex_list = self.load_json("regex_list.json")
        for regex in regex_list:
            for index, semantic_prediction_data in enumerate(semantic_predictions_data):
                text = self.get_text_from_pdf_tags(semantic_prediction_data.pdf_tags_data)
                if predictions[index]:
                    break

                matches = re.findall(regex, text)
                if matches:
                    predictions[index] = matches[0]

        return predictions
