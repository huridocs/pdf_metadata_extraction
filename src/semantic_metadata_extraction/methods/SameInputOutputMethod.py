from typing import List

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method


class SameInputOutputMethod(Method):
    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0, []

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        correct = [test for test in performance_test_set if test.text == test.segment_text]
        return 100 * len(correct) / len(performance_test_set), [x.text for x in performance_test_set]

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        return [self.clean(text) for text in texts]
