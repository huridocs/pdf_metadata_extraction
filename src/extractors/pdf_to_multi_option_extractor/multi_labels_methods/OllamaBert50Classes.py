import os
import shutil
from math import exp
from os.path import join, exists
from pathlib import Path

import pandas as pd
from transformers import TrainingArguments, AutoTokenizer

from config import ROOT_PATH
from data.Option import Option
from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.bert_method_scripts.get_batch_size import get_batch_size
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod

from extractors.bert_method_scripts.multi_label_sequence_classification_trainer import (
    multi_label_run,
    MultiLabelDataTrainingArguments,
    ModelArguments,
)

MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class OllamaBert50Classes(MultiLabelMethod):
    top_options = [
        "intellectual property",
        "telecommunication",
        "access to information",
        "privacy",
        "freedom of expression",
        "constitution",
        "electronic communications",
        "data protection and retention",
        "trademark",
        "cybercrime",
        "copyright",
        "media/press",
        "defamation",
        "data protection",
        "intermediary liability",
        "e-transactions",
        "broadcasting networks",
        "surveillance",
        "internet service providers",
        "national security",
        "penal code",
        "advertising",
        "social media",
        "e-commerce",
        "cyber security",
        "consumer protection",
        "information security",
        "access to internet",
        "freedom of press",
        "postal services",
        "digital rights",
        "terrorism",
        "censorship",
        "content regulation / censorship",
        "search engines",
        "obscenity",
        "public interest",
        "trademark law",
        "electronic media",
        "slander",
        "e-government",
        "right to be forgotten",
        "criminal procedure code",
        "domain name",
        "filtering and blocking",
        "freedom of association",
        "sedition",
        "encryption",
        "libel",
        "privacy, data protection and retention",
    ]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return extraction_data.multi_value

    def get_data_path(self, name):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, f"{name}.csv")

    def get_model_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    def get_top_one_hot_encoding(self, multi_option_data: ExtractionData):
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(self.top_options))
            for option in sample.labeled_data.values:
                if option.label not in self.top_options:
                    continue
                one_hot_encoding[-1][self.top_options.index(option.label)] = 1
        return one_hot_encoding

    @staticmethod
    def get_text(sample: TrainingSample) -> str:
        file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
        text = Path(ROOT_PATH, "data", "cyrilla_summaries", file_name).read_text()

        if "three sentence" in text.split(":")[0]:
            text = ":".join(text.split(":")[1:]).strip()

        return text if text else "No text"

    def get_top_texts_labels(self, extraction_data: ExtractionData, name: str):
        labels = self.get_top_one_hot_encoding(extraction_data)
        texts = list()
        one_hot_labels = list()

        for label, sample in zip(labels, extraction_data.samples):
            if sample.labeled_data.language_iso not in ["en", "eng"]:
                raise Exception(f"Language {sample.labeled_data.language_iso} not supported")

            if name != "predict" and sum(label) == 0:
                continue

            texts.append(self.get_text(sample))
            one_hot_labels.append(label)

        return texts, one_hot_labels

    def create_dataset(self, multi_option_data: ExtractionData, name: str):
        texts, labels = self.get_top_texts_labels(multi_option_data, name)

        rows = list()

        for text, label in zip(texts, labels):
            rows.append([text, label])

        output_df = pd.DataFrame(rows)
        output_df.columns = ["text", "labels"]

        if name != "predict":
            output_df = output_df.sample(frac=1, random_state=22).reset_index(drop=True)

        output_df.to_csv(self.get_data_path(name))
        return self.get_data_path(name)

    def train(self, multi_option_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        training_csv_path = self.create_dataset(multi_option_data, "train")
        validation_csv_path = self.create_dataset(multi_option_data, "validation")
        model_arguments = ModelArguments(MODEL_NAME)
        labels_number = len(self.top_options)

        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=training_csv_path,
            validation_file=validation_csv_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        batch_size = get_batch_size(len(multi_option_data.samples))
        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            eval_accumulation_steps=batch_size,
            learning_rate=5e-05,
            do_train=True,
            do_eval=False,
            do_predict=False,
            save_total_limit=2,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=500000,
            save_steps=200,
            load_best_model_at_end=False,
            logging_steps=50,
            metric_for_best_model="f1",
            num_train_epochs=23,
        )

        multi_label_run(model_arguments, data_training_arguments, t5_training_arguments)

    @staticmethod
    def logit_to_probabilities(logits):
        odds = [1 / (1 + exp(-logit)) for logit in logits]
        return odds

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        labels_number = len(self.top_options)
        predict_path = self.create_dataset(multi_option_data, "predict")
        model_arguments = ModelArguments(self.get_model_path(), ignore_mismatched_sizes=True)
        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=predict_path,
            validation_file=predict_path,
            test_file=predict_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        batch_size = get_batch_size(len(multi_option_data.samples))
        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            eval_accumulation_steps=batch_size,
            do_train=False,
            do_eval=False,
            do_predict=True,
        )

        logits = multi_label_run(model_arguments, data_training_arguments, t5_training_arguments)
        return self.predictions_to_options([self.logit_to_probabilities(logit) for logit in logits])

    def predictions_to_options(self, predictions) -> list[list[Option]]:
        predictions_options = list()
        options_labels = [x.label for x in self.options]
        for prediction in predictions:
            predictions_options.append([])
            for i, one_prediction in enumerate(prediction):
                if one_prediction < 0.5:
                    continue
                index = options_labels.index(self.top_options[i])
                predictions_options[-1].append(self.options[index])

        return predictions_options
