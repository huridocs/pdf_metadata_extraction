from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod


class FastSegmentSelectorMT5TrueCaseEnglishSpanishMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = MT5TrueCaseEnglishSpanishMethod