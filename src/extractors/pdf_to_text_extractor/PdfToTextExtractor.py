from data.ExtractionData import ExtractionData
from extractors.ToTextExtractor import ToTextExtractor
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod
from extractors.pdf_to_text_extractor.methods.LastDateMethod import LastDateMethod
from extractors.pdf_to_text_extractor.methods.PdfToTextRegexMethod import PdfToTextRegexMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import SegmentSelectorDateParserMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserWithBreaksMethod import \
    SegmentSelectorDateParserWithBreaksMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorInputWithoutSpaces import SegmentSelectorInputWithoutSpaces
from extractors.pdf_to_text_extractor.methods.SegmentSelectorMT5TrueCaseEnglishSpanishMethod import \
    SegmentSelectorMT5TrueCaseEnglishSpanishMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorRegexMethod import SegmentSelectorRegexMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorRegexSubtractionMethod import \
    SegmentSelectorRegexSubtractionMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorSameInputOutputMethod import \
    SegmentSelectorSameInputOutputMethod


class PdfToTextExtractor(ToTextExtractor):

    METHODS: list[type[ToTextExtractorMethod]] = [
        PdfToTextRegexMethod,
        FirstDateMethod,
        LastDateMethod,
        SegmentSelectorDateParserMethod,
        SegmentSelectorDateParserWithBreaksMethod,
        SegmentSelectorInputWithoutSpaces,
        SegmentSelectorMT5TrueCaseEnglishSpanishMethod,
        SegmentSelectorRegexMethod,
        SegmentSelectorRegexSubtractionMethod,
        SegmentSelectorSameInputOutputMethod,
    ]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False
