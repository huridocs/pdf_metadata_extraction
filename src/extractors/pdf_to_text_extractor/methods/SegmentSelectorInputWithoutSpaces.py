from extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import \
    SegmentSelectorSameInputOutputMethod
from extractors.text_to_text_extractor.methods.InputWithoutSpaces import InputWithoutSpaces


class SegmentSelectorInputWithoutSpaces(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = InputWithoutSpaces