from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod


class FastSegmentSelectorSameInputOutputMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = SameInputOutputMethod
