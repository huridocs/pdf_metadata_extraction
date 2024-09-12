from data.ExtractionData import ExtractionData
from extractors.ExtractorBase import ExtractorBase
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMethod import SetFitMethod


class ClipViTB32MultilingualMethod(SetFitMethod):
    model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False
