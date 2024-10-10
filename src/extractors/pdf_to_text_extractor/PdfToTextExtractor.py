from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.ExtractorBase import ExtractorBase
from extractors.ToTextExtractor import ToTextExtractor
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorDateParserMethod import FastSegmentSelectorDateParserMethod
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorDateParserWithBreaksMethod import (
    FastSegmentSelectorDateParserWithBreaksMethod,
)
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorMT5TrueCaseEnglishSpanishMethod import (
    FastSegmentSelectorMT5TrueCaseEnglishSpanishMethod,
)
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorRegexMethod import FastSegmentSelectorRegexMethod
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorRegexSubtractionMethod import (
    FastSegmentSelectorRegexSubtractionMethod,
)
from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorSameInputOutputMethod import (
    FastSegmentSelectorSameInputOutputMethod,
)
from extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod
from extractors.pdf_to_text_extractor.methods.LastDateMethod import LastDateMethod
from extractors.pdf_to_text_extractor.methods.PdfToTextRegexMethod import PdfToTextRegexMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import SegmentSelectorDateParserMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserWithBreaksMethod import (
    SegmentSelectorDateParserWithBreaksMethod,
)
from extractors.pdf_to_text_extractor.methods.SegmentSelectorInputWithoutSpaces import SegmentSelectorInputWithoutSpaces
from extractors.pdf_to_text_extractor.methods.SegmentSelectorMT5TrueCaseEnglishSpanishMethod import (
    SegmentSelectorMT5TrueCaseEnglishSpanishMethod,
)
from extractors.pdf_to_text_extractor.methods.SegmentSelectorRegexMethod import SegmentSelectorRegexMethod
from extractors.pdf_to_text_extractor.methods.SegmentSelectorRegexSubtractionMethod import (
    SegmentSelectorRegexSubtractionMethod,
)
from extractors.pdf_to_text_extractor.methods.SegmentSelectorSameInputOutputMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from extractors.segment_selector.FastAndPositionsSegmentSelector import FastAndPositionsSegmentSelector
from extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from extractors.segment_selector.SegmentSelector import SegmentSelector


class PdfToTextExtractor(ToTextExtractor):

    METHODS: list[type[ToTextExtractorMethod]] = [
        PdfToTextRegexMethod,
        FirstDateMethod,
        LastDateMethod,
        FastSegmentSelectorDateParserMethod,
        FastSegmentSelectorDateParserWithBreaksMethod,
        FastSegmentSelectorInputWithoutSpaces,
        FastSegmentSelectorMT5TrueCaseEnglishSpanishMethod,
        FastSegmentSelectorRegexMethod,
        FastSegmentSelectorRegexSubtractionMethod,
        FastSegmentSelectorSameInputOutputMethod,
        SegmentSelectorDateParserMethod,
        SegmentSelectorDateParserWithBreaksMethod,
        SegmentSelectorInputWithoutSpaces,
        SegmentSelectorMT5TrueCaseEnglishSpanishMethod,
        SegmentSelectorRegexMethod,
        SegmentSelectorRegexSubtractionMethod,
        SegmentSelectorSameInputOutputMethod,
    ]

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        SegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        FastSegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        FastAndPositionsSegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        return super().create_model(extraction_data)

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData) -> (ExtractionData, ExtractionData):
        if len(extraction_data.samples) < 10:
            return extraction_data, extraction_data

        samples_with_label_segments_boxes = [x for x in extraction_data.samples if x.labeled_data.label_segments_boxes]
        samples_without_label_segments_boxes = [
            x for x in extraction_data.samples if not x.labeled_data.label_segments_boxes
        ]

        train_size = int(len(samples_with_label_segments_boxes) * 0.8)
        train_set: list[TrainingSample] = (
            samples_with_label_segments_boxes[:train_size] + samples_without_label_segments_boxes
        )

        if len(extraction_data.samples) < 15:
            test_set: list[TrainingSample] = samples_with_label_segments_boxes[-10:]
        else:
            test_set = extraction_data.samples[train_size:]

        train_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, train_set)
        test_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, test_set)
        return train_extraction_data, test_extraction_data

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False
