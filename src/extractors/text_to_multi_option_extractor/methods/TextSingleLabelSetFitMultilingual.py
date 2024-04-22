from data.ExtractionData import ExtractionData
from extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import TextSingleLabelSetFit


class TextSingleLabelSetFitMultilingual(TextSingleLabelSetFit):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def train(self, extraction_data: ExtractionData):
        if self.is_multilingual(extraction_data):
            super().train(extraction_data)
