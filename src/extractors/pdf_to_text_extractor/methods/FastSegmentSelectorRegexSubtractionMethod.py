from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.RegexSubtractionMethod import RegexSubtractionMethod


class FastSegmentSelectorRegexSubtractionMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = RegexSubtractionMethod
