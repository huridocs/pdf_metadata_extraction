from data.ExtractionData import ExtractionData
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMethod import SingleLabelSetFitMethod


class SingleLabelSetFitMultilingualMethod(SingleLabelSetFitMethod):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if extraction_data.multi_value:
            return False

        if self.is_multilingual(extraction_data):
            return True

        return False
