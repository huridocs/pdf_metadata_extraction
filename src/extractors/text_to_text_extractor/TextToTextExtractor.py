import json
import os
from pathlib import Path

from config import config_logger
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.ToTextExtractor import ToTextExtractor
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod
from extractors.text_to_text_extractor.methods.InputWithoutSpaces import InputWithoutSpaces
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from extractors.text_to_text_extractor.methods.RegexSubtractionMethod import RegexSubtractionMethod
from extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod
from send_logs import send_logs


class TextToTextExtractor(ToTextExtractor):
    METHODS: list[type[ToTextExtractorMethod]] = [
        SameInputOutputMethod,
        InputWithoutSpaces,
        RegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.tags_texts:
                return True

        return False
