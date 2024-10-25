from data.ExtractionData import ExtractionData
from extractors.ToTextExtractor import ToTextExtractor
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod
from extractors.text_to_text_extractor.methods.GlinerDateParserMethod import GlinerDateParserMethod
from extractors.text_to_text_extractor.methods.InputWithoutSpaces import InputWithoutSpaces
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import NerFirstAppearanceMethod
from extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import NerLastAppearanceMethod
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from extractors.text_to_text_extractor.methods.RegexSubtractionMethod import RegexSubtractionMethod
from extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod


class TextToTextExtractor(ToTextExtractor):
    METHODS: list[type[ToTextExtractorMethod]] = [
        SameInputOutputMethod,
        InputWithoutSpaces,
        RegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        GlinerDateParserMethod,
        NerFirstAppearanceMethod,
        NerLastAppearanceMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.tags_texts:
                return True

        return False
