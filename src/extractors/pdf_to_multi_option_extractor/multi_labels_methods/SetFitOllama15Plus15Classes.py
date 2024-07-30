import os
import shutil
from os.path import join, exists
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from config import ROOT_PATH
from data.ExtractionData import ExtractionData
from data.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from data.TrainingSample import TrainingSample
from extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import EarlyStoppingAfterInitialTraining
from extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod


class SetFitOllama15Plus15Classes(MultiLabelMethod):
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

    top_options = [
        [
            "intellectual property",
            "telecommunication",
            "access to information",
            "freedom of expression",
            "privacy",
            "constitution",
            "trademark",
            "electronic communications",
            "cybercrime",
            "data protection and retention",
            "data protection",
            "media/press",
            "cybercrime",
            "copyright",
            "penal code",
        ],
        [
            "defamation",
            "surveillance",
            "e-transactions",
            "digital rights",
            "social media",
            "libel",
            "access to internet",
            "broadcasting networks",
            "terrorism",
            "advertising",
            "right to be forgotten",
            "information security",
            "freedom of press",
            "public interest",
            "internet service providers",
        ],
    ]

    def get_data_path(self, model_index: int):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, f"data_{model_index}.csv")

    def get_model_path(self, model_index: int):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "setfit_model_" + str(model_index))

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    @staticmethod
    def get_text(sample: TrainingSample) -> str:
        file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
        text = Path(ROOT_PATH, "data", "cyrilla_summaries", file_name).read_text()

        if "three sentence" in text.split(":")[0]:
            text = ":".join(text.split(":")[1:]).strip()

        return text if text else "No text"

    def get_top_one_hot_encoding(self, multi_option_data: ExtractionData, model_index: int):
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(self.top_options[model_index]))
            for option in sample.labeled_data.values:
                if option.label not in self.top_options[model_index]:
                    continue
                one_hot_encoding[-1][self.top_options[model_index].index(option.label)] = 1
        return one_hot_encoding

    def get_dataset_from_data(self, extraction_data: ExtractionData, model_index: int):
        labels = self.get_top_one_hot_encoding(extraction_data, model_index)
        data = list()

        for label, sample in zip(labels, extraction_data.samples):
            if sample.labeled_data.language_iso not in ["en", "eng"]:
                raise Exception(f"Language {sample.labeled_data.language_iso} not supported")

            if sum(label) == 0:
                continue

            data.append([self.get_text(sample), label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(str(self.get_data_path(model_index)))
        dataset_csv = load_dataset("csv", data_files=str(self.get_data_path(model_index)))
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, extraction_data: ExtractionData):
        for model_index in range(2):
            shutil.rmtree(self.get_model_path(model_index), ignore_errors=True)
            train_dataset = self.get_dataset_from_data(extraction_data, model_index)
            batch_size = get_batch_size(len(extraction_data.samples))

            model = SetFitModel.from_pretrained(
                self.model_name,
                labels=self.top_options[model_index],
                multi_target_strategy="one-vs-rest",
                use_differentiable_head=True,
                head_params={"out_features": len(self.top_options[model_index])},
            )

            args = TrainingArguments(
                output_dir=self.get_model_path(model_index),
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

            trainer.model.save_pretrained(self.get_model_path(model_index))

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        all_predictions = [list() for _ in multi_option_data.samples]
        for model_index in range(2):
            model = SetFitModel.from_pretrained(self.get_model_path(model_index))
            predict_texts = [self.get_text(sample) for sample in multi_option_data.samples]

            predictions = model.predict(predict_texts)

            predictions_options = self.predictions_to_options(predictions.tolist(), model_index)
            for prediction, prediction_model in zip(all_predictions, predictions_options):
                prediction.extend(prediction_model)

        return all_predictions

    def predictions_to_options(self, predictions, model_index: int) -> list[list[Option]]:
        predictions_options = list()
        options_labels = [x.label for x in self.options]
        for prediction in predictions:
            predictions_options.append([])
            for i, one_prediction in enumerate(prediction):
                if one_prediction < 0.5:
                    continue
                index = options_labels.index(self.top_options[model_index][i])
                predictions_options[-1].append(self.options[index])

        return predictions_options

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        return True
