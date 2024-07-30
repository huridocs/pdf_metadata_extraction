import json
import os
import shutil
from collections import Counter
from os.path import join, exists
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from numpy import argmax

from data.ExtractionData import ExtractionData
from data.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import EarlyStoppingAfterInitialTraining
from extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod


class SetFitOptionsWithSamplesMethod3(MultiLabelMethod):
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    THRESHOLD = 3

    def get_data_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "setfit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    @staticmethod
    def get_one_hot_encoding_with_samples(multi_option_data: ExtractionData, options_with_samples: list[Option]):
        options_ids = [option.id for option in options_with_samples]
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(options_ids))
            for option in sample.labeled_data.values:
                if option.id not in options_ids:
                    continue
                one_hot_encoding[-1][options_ids.index(option.id)] = 1
        return one_hot_encoding

    def get_dataset_from_data(self, extraction_data: ExtractionData, options_with_samples: list[Option]):
        data = list()
        texts = [sample.pdf_data.get_text() for sample in extraction_data.samples]
        labels = self.get_one_hot_encoding_with_samples(extraction_data, options_with_samples)

        for text, label, sample in zip(texts, labels, extraction_data.samples):
            if sample.labeled_data.language_iso not in ["en", "eng"]:
                continue

            if sum(label) == 0:
                continue

            data.append([text, label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, extraction_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)
        train_dataset = self.get_dataset_from_data(extraction_data, self.get_options_with_samples(extraction_data))
        batch_size = get_batch_size(len(extraction_data.samples))

        model = SetFitModel.from_pretrained(
            self.model_name,
            labels=[x.label for x in self.options],
            multi_target_strategy="one-vs-rest",
        )

        args = TrainingArguments(
            output_dir=self.get_model_path(),
            batch_size=batch_size,
            max_steps=get_max_steps(len(extraction_data.samples)),
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            metric="f1",
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidAllEvaluation()],
        )

        trainer.train()

        trainer.model.save_pretrained(self.get_model_path())

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        model = SetFitModel.from_pretrained(self.get_model_path())
        predict_texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]
        predictions = model.predict(predict_texts)

        return self.predictions_to_options(predictions.tolist())

    def predictions_to_options(self, predictions) -> list[list[Option]]:
        options_with_samples = json.loads(self.get_options_with_samples_path().read_text())

        one_hot_list = list()
        for prediction in predictions:
            if not self.multi_value:
                best_score_index = argmax(prediction)

                if prediction[best_score_index] > 0.5:
                    one_hot_list.append([])
                else:
                    best_label = options_with_samples[best_score_index]
                    one_hot_list.append([[self.label_to_option(best_label)]])

            one_hot_list.append(
                [self.label_to_option(options_with_samples[i]) for i, value in enumerate(prediction) if value > 0.5]
            )

        return one_hot_list

    def label_to_option(self, label: str) -> Option:
        return [x for x in self.options if x.label == label][0]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        return True

    def get_options_with_samples_path(self):
        return Path(join(self.extraction_identifier.get_path(), str(self.get_name()), "options_with_samples.json"))

    def get_options_with_samples(self, extraction_data) -> list[Option]:
        amount_per_option = Counter()
        for sample in [x for x in extraction_data.samples if x.labeled_data.language_iso in ["en", "eng"]]:
            amount_per_option.update([x.label for x in sample.labeled_data.values])

        options_with_samples = [option for option in self.options if amount_per_option[option.label] >= self.THRESHOLD]
        self.get_options_with_samples_path().write_text(json.dumps([x.label for x in options_with_samples]))
        return options_with_samples
