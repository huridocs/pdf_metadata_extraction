from extractors.pdf_to_text_extractor.methods.SegmentSelectorSameInputOutputMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod


class SegmentSelectorDateParserMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = DateParserMethod
