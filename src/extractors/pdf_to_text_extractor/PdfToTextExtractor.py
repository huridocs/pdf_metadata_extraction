from data.ExtractionData import ExtractionData
from extractors.ToTextExtractor import ToTextExtractor
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorSemanticExtractorMethod import \
    SegmentSelectorSemanticExtractorMethod


class PdfToTextExtractor(ToTextExtractor):

    METHODS: list[type[ToTextExtractorMethod]] = [SegmentSelectorSemanticExtractorMethod]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False



