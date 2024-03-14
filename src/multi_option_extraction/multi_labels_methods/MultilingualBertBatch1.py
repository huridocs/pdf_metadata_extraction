import os
import shutil
from math import exp
from os.path import join, exists

import pandas as pd
from transformers import TrainingArguments

from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.TextToMultiOptionMethod import MultiLabelMethods

from multi_option_extraction.text_to_multi_option_methods.multi_label_sequence_classification_trainer import (
    multi_label_run,
    MultiLabelDataTrainingArguments,
    ModelArguments,
)

MODEL_NAME = "google-bert/bert-base-multilingual-cased"


class MultilingualBertBatch1(MultiLabelMethods):
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

    def create_dataset(self, multi_option_data, name: str):
        pdf_tags = [x.pdf_data for x in multi_option_data.samples]
        texts = [self.get_text_from_pdf_segments(x) for x in pdf_tags]
        labels = self.get_one_hot_encoding(multi_option_data)

        rows = list()

        for text, label in zip(texts, labels):
            rows.append([text, label])

        output_df = pd.DataFrame(rows)
        output_df.columns = ["text", "labels"]
        output_df = output_df.sample(frac=1, random_state=22).reset_index(drop=True)

        output_df.to_csv(self.get_data_path(name))
        return self.get_data_path(name)

    def train(self, multi_option_data: MultiOptionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        training_csv_path = self.create_dataset(multi_option_data, "train")
        validation_csv_path = self.create_dataset(multi_option_data, "validation")
        model_arguments = ModelArguments(MODEL_NAME)
        labels_number = len(self.options)

        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=training_csv_path,
            validation_file=validation_csv_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            bf16=False,
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

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[list[Option]]:
        labels_number = len(self.options)
        test_path = self.get_predict_dataframe(semantic_predictions_data)
        model_arguments = ModelArguments(self.get_model_path(), ignore_mismatched_sizes=True)
        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=test_path,
            validation_file=test_path,
            test_file=test_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            do_train=False,
            do_eval=False,
            do_predict=True,
        )

        logits = multi_label_run(model_arguments, data_training_arguments, t5_training_arguments)
        return self.predictions_to_options_list([self.logit_to_probabilities(logit) for logit in logits])

    def get_predict_dataframe(self, semantic_predictions_data: list[SemanticPredictionData]):
        pdf_tags = [x.pdf_tags_data for x in semantic_predictions_data]
        texts = [self.get_text_from_pdf_segments(x) for x in pdf_tags]
        labels_number = len(self.options)
        output_df = pd.DataFrame([[text, [0] * labels_number] for text in texts])
        output_df.columns = ["text", "labels"]
        output_df.to_csv(self.get_data_path("predict"))
        test_path = self.get_data_path("predict")
        return test_path
