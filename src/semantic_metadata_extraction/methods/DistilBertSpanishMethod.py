from typing import List

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from transformers import pipeline


class DistilBertSpanishMethod(Method):
    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        predictions = self.predict([x.segment_text for x in performance_test_set])

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set)

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        nlp = pipeline(
            "question-answering",
            model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            tokenizer=("mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", {"use_fast": False}),
        )

        predict_text = [{"question": f"¿{self.property_name.replace('_', ' ')}?", "context": text} for text in texts]
        # predict_text = [{"question": f"¿País?", "context": text} for text in texts]

        results = nlp(predict_text)
        predictions = [res["answer"] for res in results]
        return predictions
