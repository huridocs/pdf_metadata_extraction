from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase


class NaiveExtractor(ExtractorBase):
    def create_model(self, extraction_data):
        pass

    def get_suggestions(self, predictions_samples):
        suggestions = list()
        for prediction_sample in predictions_samples:
            suggestion = Suggestion.get_empty(self.extraction_identifier, prediction_sample.pdf_data.file_name)
            suggestion.text = prediction_sample.get_text()
            suggestions.append(suggestion)

        return suggestions

    def exists_model(self):
        return True
