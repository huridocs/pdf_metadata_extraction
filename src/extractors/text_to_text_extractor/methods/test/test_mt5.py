from time import time
from unittest import TestCase

import torch

from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod


class TestMT5(TestCase):
    def test_train(self):
        start = time()
        print("GPU available?")
        print(torch.cuda.is_available())
        extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")
        mt5_true_case_english_spanish = MT5TrueCaseEnglishSpanishMethod(extraction_identifier)

        semantic_information_data = [
            SemanticExtractionData(text="foo", pdf_tags=[PdfTagData.from_text("1/ foo end")], language_iso="")
        ] * 5
        semantic_information_data += [
            SemanticExtractionData(text="var", pdf_tags=[PdfTagData.from_text("2/ var end")], language_iso="")
        ] * 5

        mt5_true_case_english_spanish.train(semantic_information_data)

        predictions = mt5_true_case_english_spanish.predict([SemanticPredictionData.from_text("1/ foo end")] * 10)
        self.assertEqual(10, len(predictions))
        self.assertEqual("foo", predictions[0])
        print(f"Predictions in {round(time() - start, 2)} seconds")
