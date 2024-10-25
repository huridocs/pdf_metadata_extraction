from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.GlinerDateParserMethod import GlinerDateParserMethod


class FastSegmentSelectorGlinerDateParserMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = GlinerDateParserMethod
