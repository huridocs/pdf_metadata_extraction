import os
from functools import lru_cache
from os.path import join, exists, dirname
from typing import List
from huggingface_hub import hf_hub_download
import pandas as pd
import csv

from transformers import AutoTokenizer

from ServiceConfig import ServiceConfig
from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method

from semantic_metadata_extraction.methods.TrueCaser import TrueCaser
from semantic_metadata_extraction.methods.run_seq_2_seq import (
    ModelArguments,
    DataTrainingArguments,
    run,
    T5TrainingArguments,
)


class MT5TrueCaseEnglishSpanishMethod(Method):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    SERVICE_CONFIG = ServiceConfig()

    def get_model_path(self):
        return join(self.base_path, self.get_name(), "model")

    def get_max_input_length(self, semantic_extraction_data: List[SemanticExtractionData]):
        tokenizer = AutoTokenizer.from_pretrained("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())

        texts = [self.property_name + ": " + x.segment_text for x in semantic_extraction_data]
        tokens_number = [len(tokenizer(text)["input_ids"]) for text in texts]
        return min(int((max(tokens_number) + 5) * 1.2), 512)

    def get_max_output_length(self, semantic_extraction_data: List[SemanticExtractionData]):
        tokenizer = AutoTokenizer.from_pretrained("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())

        texts = [x.text for x in semantic_extraction_data]
        tokens_number = [len(tokenizer(text)["input_ids"]) for text in texts]
        return min(int((max(tokens_number) + 5) * 1.2), 256)

    def prepare_dataset(self, semantic_extractions_data: List[SemanticExtractionData]):
        data_path = join(self.base_path, self.get_name(), "t5_transformers_data.csv")

        if exists(data_path):
            os.remove(data_path)

        true_case_inputs = [
            self.get_true_case_segment_text(semantic_extraction_data)
            for semantic_extraction_data in semantic_extractions_data
        ]

        data = [
            [str(index), f"{self.property_name}: {segment_text}", semantic_data.text]
            for index, segment_text, semantic_data in zip(
                range(len(true_case_inputs)), true_case_inputs, semantic_extractions_data
            )
        ]
        df = pd.DataFrame(data)
        df.columns = ["id", "input_with_prefix", "target"]
        df["not_used"] = ""

        os.makedirs(dirname(data_path), exist_ok=True)
        df.to_csv(data_path, quoting=csv.QUOTE_ALL)
        return data_path

    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)
        self.train(performance_train_set)
        predictions = self.predict([x.segment_text for x in performance_test_set])

        self.remove_model()

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        self.remove_model()
        train_path = self.prepare_dataset(semantic_extraction_data)
        model_arguments = ModelArguments("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())
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
            learning_rate=5e-4,
            weight_decay=0.1,
            do_train=True,
            do_eval=True,
            do_predict=False,
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

    def get_cache_dir(self):
        cache_dir = join(self.service_config.docker_volume_path, "HF_cache")
        return cache_dir

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

    @lru_cache(maxsize=1)
    def get_true_case(self):
        true_case_english_model_path = hf_hub_download(
            repo_id="HURIDOCS/spanish-truecasing",
            filename="english.dist",
            revision="69558da13436cc2b29d7db92d704976e2a7ffe16",
            cache_dir=self.service_config.docker_volume_path,
        )

        true_case_spanish_model_path = hf_hub_download(
            repo_id="HURIDOCS/spanish-truecasing",
            filename="spanish.dist",
            revision="69558da13436cc2b29d7db92d704976e2a7ffe16",
            cache_dir=self.service_config.docker_volume_path,
        )

        os.makedirs(join(self.service_config.docker_volume_path, "nltk_data"), exist_ok=True)
        return TrueCaser(true_case_english_model_path), TrueCaser(true_case_spanish_model_path)

    def get_true_case_segment_text(self, semantic_extraction_data):
        true_case_english, true_case_spanish = self.get_true_case()

        if semantic_extraction_data.language_iso == "en":
            return true_case_english.get_true_case(semantic_extraction_data.segment_text)

        if semantic_extraction_data.language_iso == "es":
            return true_case_spanish.get_true_case(semantic_extraction_data.segment_text)

        return semantic_extraction_data.segment_text
