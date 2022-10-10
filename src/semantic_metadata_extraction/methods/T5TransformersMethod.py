import os
import shutil
from os.path import join, exists, basename
from typing import List

import pandas as pd
import csv
from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
import sentencepiece

from semantic_metadata_extraction.methods.run_seq_2_seq import (
    ModelArguments,
    DataTrainingArguments,
    run,
    T5TrainingArguments,
)


class T5TransformersMethod(Method):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    ENGLISH_SENTENCE_PIECE = f"{SCRIPT_PATH}/t5_small_spiece.model"

    def get_model_path(self):
        return join(self.base_path, basename(__file__).split(".")[0])

    def get_max_input_length(self, semantic_extraction_data: List[SemanticExtractionData]):
        sentence_piece = sentencepiece.SentencePieceProcessor(self.ENGLISH_SENTENCE_PIECE)

        texts = [self.property_name + ": " + x.segment_text for x in semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return min(int((max(tokens_number) + 1) * 1.2), 512)

    def get_max_output_length(self, semantic_extraction_data: List[SemanticExtractionData]):
        sentence_piece = sentencepiece.SentencePieceProcessor(self.ENGLISH_SENTENCE_PIECE)

        texts = [x.text for x in semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return min(int((max(tokens_number) + 1) * 1.2), 256)

    @staticmethod
    def get_batch_size(semantic_extraction_data: List[SemanticExtractionData]):
        if len(semantic_extraction_data) > 16:
            return 8

        return 1

    def prepare_dataset(self, semantic_extraction_data: List[SemanticExtractionData]):
        data_path = join(self.base_path, "t5_transformers_data.csv")

        if exists(data_path):
            os.remove(data_path)

        data = [
            [str(index), f"{self.property_name}: {x.segment_text}", x.text]
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
        self.train(performance_train_set)
        predictions = self.predict([x.segment_text for x in performance_test_set])
        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set)

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        self.remove_model_if_exists()
        train_path = self.prepare_dataset(semantic_extraction_data)

        model_arguments = ModelArguments("t5-small")
        length = self.get_max_output_length(semantic_extraction_data)
        data_training_arguments = DataTrainingArguments(
            train_file=train_path,
            validation_file=train_path,
            max_seq_length=self.get_max_input_length(semantic_extraction_data),
            max_answer_length=length,
            context_column="input_with_prefix",
            answer_column="target",
            question_column="not_used",
            version_2_with_negative=False,
        )

        t5_training_arguments = T5TrainingArguments(
            report_to=[],
            predict_with_generate=True,
            overwrite_output_dir=True,
            output_dir=self.get_model_path(),
            per_device_train_batch_size=1,
            weight_decay=0.1,
            # learning_rate=3e-5,
            learning_rate=5e-4,
            do_train=True,
            do_eval=True,
            do_predict=False,
            eval_accumulation_steps=1,
            save_total_limit=2,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            early_stopping=True,
            num_train_epochs=30,
            early_stopping_patience=4,
            generation_max_length=self.get_max_output_length(semantic_extraction_data),
        )

        run(model_arguments, data_training_arguments, t5_training_arguments)

    def exists_model(self):
        return exists(self.get_model_path())

    def predict(self, texts: List[str]) -> List[str]:
        if not self.exists_model():
            return texts

        semantic_extraction_data = [
            SemanticExtractionData(text="predict", segment_text=text, language_iso="en") for text in texts
        ]
        predict_data_path = self.prepare_dataset(semantic_extraction_data)

        model_arguments = ModelArguments(self.get_model_path())
        data_training_arguments = DataTrainingArguments(
            test_file=predict_data_path,
            context_column="input_with_prefix",
            answer_column="target",
            question_column="not_used",
            version_2_with_negative=False,
        )
        seq_seq_training_arguments = T5TrainingArguments(
            report_to=[],
            predict_with_generate=True,
            output_dir=self.get_model_path(),
            do_train=False,
            do_eval=False,
            do_predict=True,
            generation_max_length=self.get_max_output_length(semantic_extraction_data),
        )

        predictions = run(model_arguments, data_training_arguments, seq_seq_training_arguments)
        return [prediction["prediction_text"] for prediction in predictions]

    def remove_model_if_exists(self):
        if self.exists_model():
            shutil.rmtree(self.get_model_path(), ignore_errors=True)