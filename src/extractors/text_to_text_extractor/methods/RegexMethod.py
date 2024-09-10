import re

from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample

from tdda import *

from extractors.ToTextExtractorMethod import ToTextExtractorMethod


class RegexMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        regex_list = rexpy.extract([x.labeled_data.label_text for x in extraction_data.samples])
        regex_list = [regex[1:-1] for regex in regex_list]
        self.save_json("regex_list.json", regex_list)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        predictions = ["" for _ in predictions_samples]
        regex_list = self.load_json("regex_list.json")
        for regex in regex_list:
            for index, prediction_sample in enumerate(predictions_samples):
                text = " ".join(prediction_sample.tags_texts)
                if predictions[index]:
                    break

                matches = re.findall(regex, text)
                if matches:
                    predictions[index] = matches[0]

        return predictions
