import os
import shutil
from os.path import join, exists, basename
from typing import List

import pandas as pd
import csv
from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from transformers import AutoModelWithLMHead, AutoTokenizer


class T5ZeroShot(Method):

    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    def get_model_path(self):
        return join(self.base_path, basename(__file__).split(".")[0])

    def prepare_dataset(self, semantic_extraction_data: List[SemanticExtractionData]):
        data_path = join(self.base_path, "t5_zero_shot.csv")

        if exists(data_path):
            os.remove(data_path)

        property_name = self.property_name.replace("_", " ")
        data = [
            [str(index), f"question: {property_name}?  context: {x.segment_text} </s>", x.text]
            for index, x in enumerate(semantic_extraction_data)
        ]
        df = pd.DataFrame(data)
        df.columns = ["id", "input_with_prefix", "target"]
        df["not_used"] = ""

        df.to_csv(data_path, quoting=csv.QUOTE_ALL)
        return data_path

    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)
        predictions = self.predict([x.segment_text for x in performance_test_set])
        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        semantic_extraction_data = [
            SemanticExtractionData(text="predict", segment_text=text, language_iso="en") for text in texts
        ]
        predict_data_path = self.prepare_dataset(semantic_extraction_data)

        model = "valhalla/t5-base-squad"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelWithLMHead.from_pretrained(model)

        def get_answer(input_text):
            input_text = f"{input_text} </s>"
            features = tokenizer([input_text], return_tensors="pt")

            out = model.generate(input_ids=features["input_ids"], attention_mask=features["attention_mask"])

            return tokenizer.decode(out[0])

        df = pd.read_csv(predict_data_path)
        predictions = list()

        for index, input_text in enumerate(df["input_with_prefix"].tolist()):
            predictions.append(get_answer(input_text))

        predictions = [str(p)[6:-4] for p in predictions]

        return predictions

    def remove_model_if_exists(self):
        if self.exists_model():
            shutil.rmtree(self.get_model_path(), ignore_errors=True)
