from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import (
    NerLastAppearanceMethod,
)

from config import FLAIR_NER_METHODS_ENABLED

# The Flair based NER methods are disabled while the installed Flair/torch combination cannot
# load the `ner-ontonotes-large` checkpoint. Their code stays in trainable_entity_extractor; we
# only keep them out of the registered METHODS lists so training does not attempt to load the
# model. Re-enable with the FLAIR_NER_METHODS_ENABLED environment variable.
DISABLED_TEXT_TO_TEXT_METHODS = [NerFirstAppearanceMethod, NerLastAppearanceMethod]


def _disable_flair_ner_methods() -> None:
    for extractor in (TextToTextExtractor, PdfToTextExtractor):
        extractor.METHODS = [
            method
            for method in extractor.METHODS
            if getattr(method, "SEMANTIC_METHOD", method) not in DISABLED_TEXT_TO_TEXT_METHODS
        ]


if not FLAIR_NER_METHODS_ENABLED:
    _disable_flair_ner_methods()


EXTRACTORS = [
    PdfToMultiOptionExtractor,
    TextToMultiOptionExtractor,
    PdfToTextExtractor,
    TextToTextExtractor,
]
