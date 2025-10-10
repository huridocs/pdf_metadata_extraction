from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor

EXTRACTORS = [
    PdfToMultiOptionExtractor,
    TextToMultiOptionExtractor,
    PdfToTextExtractor,
    TextToTextExtractor,
]
