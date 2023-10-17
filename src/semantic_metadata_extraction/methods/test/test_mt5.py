from time import time
from unittest import TestCase

import torch

from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod


class TestMT5(TestCase):
    def test_train(self):
        # pass
        start = time()
        print("GPU available?")
        print(torch.cuda.is_available())

        mt5_true_case_english_spanish = MT5TrueCaseEnglishSpanishMethod("mt5_test", "mt5_test")

        semantic_information_data = [
            SemanticExtractionData(text="foo", pdf_tags=[PdfTagData.from_text("1/ foo end")], language_iso="")
        ] * 10
        semantic_information_data += [
            SemanticExtractionData(text="var", pdf_tags=[PdfTagData.from_text("2/ var end")], language_iso="")
        ] * 10

        mt5_true_case_english_spanish.train(semantic_information_data)

        predictions = mt5_true_case_english_spanish.predict([SemanticPredictionData.from_text("1/ foo end")] * 300)
        self.assertEqual(300, len(predictions))
        self.assertEqual("foo", predictions[0])
        print(f"Predictions in {round(time() - start, 2)} seconds")
