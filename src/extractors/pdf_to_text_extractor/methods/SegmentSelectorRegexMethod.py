from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import \
    SegmentSelectorSameInputOutputMethod
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod


class SegmentSelectorRegexMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = RegexMethod