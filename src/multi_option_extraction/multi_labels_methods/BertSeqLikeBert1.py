import os
import shutil
from math import exp
from os.path import join, exists
import evaluate
import numpy as np
import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.text_to_multi_option_methods.AvoidEvaluation import AvoidEvaluation
from multi_option_extraction.text_to_multi_option_methods.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.TextToMultiOptionMethod import MultiLabelMethods

MODEL_NAME = "google-bert/bert-base-uncased"

clf_metrics = evaluate.combine(["accuracy"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BertSeqLikeBert1(MultiLabelMethods):
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

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = 1 / (1 + np.exp(-predictions))
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))

    def preprocess_function(self, multi_option_sample: MultiOptionSample):
        text = multi_option_sample.get_text()
        labels = [1.0 if value in multi_option_sample.values else 0.0 for value in self.options]

        example = tokenizer(text, padding="max_length", truncation="only_first", max_length=self.get_token_length())
        example["labels"] = labels
        return example

    def train(self, multi_option_data: MultiOptionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        self.create_dataset(multi_option_data, "train")

        examples = [self.preprocess_function(x) for x in multi_option_data.samples]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        id2class = {index: label for index, label in enumerate([x.label for x in self.options])}
        class2id = {label: index for index, label in enumerate([x.label for x in self.options])}

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.options),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification",
        )

        training_args = TrainingArguments(
            output_dir=self.get_model_path(),
            learning_rate=5e-05,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.get_batch_size(multi_option_data),
            per_device_eval_batch_size=self.get_batch_size(multi_option_data),
            max_steps=1219,
            weight_decay=0.01,
            eval_steps=53,
            save_steps=53,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            fp16=False,
            bf16=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=examples,
            eval_dataset=examples,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidEvaluation()],
        )

        trainer.train()

        trainer.save_model(self.get_model_path())

    @staticmethod
    def logit_to_probabilities(logits):
        odds = [1 / (1 + exp(-logit)) for logit in logits]
        return odds

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[list[Option]]:
        id2class = {index: label for index, label in enumerate([x.label for x in self.options])}
        class2id = {label: index for index, label in enumerate([x.label for x in self.options])}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.get_model_path(),
            num_labels=len(self.options),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification",
        )

        model.eval()

        inputs = tokenizer(
            [x.get_text() for x in semantic_predictions_data],
            return_tensors="pt",
            padding="max_length",
            truncation="only_first",
            max_length=self.get_token_length(),
        )
        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

        return self.predictions_to_options_list([self.logit_to_probabilities(logit) for logit in output.logits])

    def get_predict_dataframe(self, semantic_predictions_data: list[SemanticPredictionData]):
        pdf_tags = [x.pdf_tags_data for x in semantic_predictions_data]
        texts = [self.get_text_from_pdf_segments(x) for x in pdf_tags]
        labels_number = len(self.options)
        output_df = pd.DataFrame([[text, [0] * labels_number] for text in texts])
        output_df.columns = ["text", "labels"]
        output_df.to_csv(self.get_data_path("predict"))
        test_path = self.get_data_path("predict")
        return test_path

    def get_token_length(self):
        data = pd.read_csv(self.get_data_path("train"))
        max_length = 0
        for index, row in data.iterrows():
            length = len(tokenizer(row["text"]).data["input_ids"])
            max_length = max(length, max_length)

        return max_length
