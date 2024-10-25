from extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import NerFirstAppearanceMethod


class FastSegmentSelectorNerFirstAppearanceMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = NerFirstAppearanceMethod
