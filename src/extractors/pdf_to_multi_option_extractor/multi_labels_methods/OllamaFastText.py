import os
import shutil
from os.path import join, exists
from pathlib import Path

import fasttext

from config import ROOT_PATH
from data.Option import Option
from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod


class OllamaFastText(MultiLabelMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    @staticmethod
    def clean_label(label: str):
        return "_".join(label.split()).lower().replace(",", "")

    def clean_labels(self, options: list[Option]):
        return [self.clean_label(option.label) for option in options]

    def get_data_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.txt")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "fast.model")

    @staticmethod
    def get_text(sample: TrainingSample) -> str:
        file_name = sample.pdf_data.pdf_features.file_name.replace('.pdf', '.txt')
        text = Path(ROOT_PATH, 'data', 'cyrilla_summaries', file_name).read_text()

        if 'three sentence' in text.split(':')[0]:
            text = ':'.join(text.split(':')[1:]).strip()

        return text if text else "No text"

    def prepare_data(self, multi_option_data: ExtractionData):
        texts = [self.get_text(sample) for sample in multi_option_data.samples]
        texts = [text.replace("\n", " ") for text in texts]
        labels = [
            "__label__" + " __label__".join(self.clean_labels(sample.labeled_data.values))
            for sample in multi_option_data.samples
        ]
        data = [f"{label} {text}" for label, text in zip(labels, texts)]
        Path(self.get_data_path()).write_text("\n".join(data))

    def train(self, multi_option_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)
        self.prepare_data(multi_option_data)
        fasttext_params = {
            "input": self.get_data_path(),
            "lr": 0.1,
            "lrUpdateRate": 1000,
            "thread": 8,
            "epoch": 600,
            "wordNgrams": 2,
            "dim": 100,
            "loss": "ova",
        }
        model = fasttext.train_supervised(**fasttext_params)
        model.save_model(self.get_model_path())

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        texts = [self.get_text(sample) for sample in multi_option_data.samples]
        texts = [text.replace("\n", " ") for text in texts]

        model = fasttext.load_model(self.get_model_path())
        labels = self.clean_labels(self.options)

        if self.multi_value:
            prediction_labels_scores = model.predict(texts, k=len(labels))
        else:
            prediction_labels_scores = model.predict(texts, k=1)

        predictions: list[list[Option]] = list()
        for prediction_labels, scores in zip(prediction_labels_scores[0], prediction_labels_scores[1]):
            predictions.append(list())
            for prediction_label, score in zip(prediction_labels, scores):
                if score > 0.5 and prediction_label[9:] in labels:
                    label_index = labels.index(prediction_label[9:])
                    predictions[-1].append(self.options[label_index])

        return predictions
