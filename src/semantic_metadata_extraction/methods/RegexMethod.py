import re
from typing import List

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method

from tdda import *


class RegexMethod(Method):
    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        self.train(performance_train_set)
        predictions = self.predict([x.segment_text for x in performance_test_set])
        self.log_performance_sample(semantic_extractions_data=performance_test_set, predictions=predictions)
        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        regex_list = rexpy.extract([x.text for x in semantic_extraction_data])
        regex_list = [regex[1:-1] for regex in regex_list]
        self.save_json("regex_list.json", regex_list)

    def predict(self, texts: List[str]) -> List[str]:
        predictions = ["" for _ in texts]
        regex_list = self.load_json("regex_list.json")
        for regex in regex_list:
            for index, text in enumerate(texts):
                if predictions[index]:
                    break

                matches = re.findall(regex, text)
                if matches:
                    predictions[index] = matches[0]

        return predictions
