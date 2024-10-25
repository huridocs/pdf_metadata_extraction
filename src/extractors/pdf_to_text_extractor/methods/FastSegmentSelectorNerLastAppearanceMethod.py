from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import NerLastAppearanceMethod


class FastSegmentSelectorNerLastAppearanceMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = NerLastAppearanceMethod
