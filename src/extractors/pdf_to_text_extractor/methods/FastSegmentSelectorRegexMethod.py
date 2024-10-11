from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod


class FastSegmentSelectorRegexMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = RegexMethod
