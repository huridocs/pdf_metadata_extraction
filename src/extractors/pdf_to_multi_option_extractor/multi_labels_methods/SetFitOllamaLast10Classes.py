import json
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


class SetFitOllamaLast10Classes(MultiLabelMethod):
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

    top_options = [
        "decryption",
        "state of emergency",
        "open data",
        "instagram community guidelines>hate speech, bullying and abuse",
        "military justice code",
        "privacy policy",
        "cyber security / cyber crime",
        "gambling",
        "insult to foreign state",
        "cybercafe"]

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
    def get_text(sample: TrainingSample) -> str:
        file_name = sample.pdf_data.pdf_features.file_name.replace('.pdf', '.txt')
        text = Path(ROOT_PATH, 'data', 'cyrilla_summaries', file_name).read_text()

        if 'three sentence' in text.split(':')[0]:
            text = ':'.join(text.split(':')[1:]).strip()

        return text if text else "No text"

    def get_top_one_hot_encoding(self, multi_option_data: ExtractionData):
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(self.top_options))
            for option in sample.labeled_data.values:
                if option.label not in self.top_options:
                    continue
                one_hot_encoding[-1][self.top_options.index(option.label)] = 1
        return one_hot_encoding

    def get_dataset_from_data(self):
        data = list()

        for label in os.listdir(Path(ROOT_PATH, 'data', 'cyrilla_synthetic_data')):
            option_label = label.replace('.txt', '').replace('___', '/')
            if option_label not in self.top_options:
                continue
            texts = json.loads(Path(ROOT_PATH, 'data', 'cyrilla_synthetic_data', label).read_text())
            for text in texts:
                one_hot = [0] * len(self.top_options)
                if option_label in self.top_options:
                    one_hot[self.top_options.index(option_label)] = 1
                else:
                    raise Exception(f"Label {label} not in {self.top_options}")
                data.append([text, one_hot])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, extraction_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)
        train_dataset = self.get_dataset_from_data()
        batch_size = get_batch_size(len(extraction_data.samples))

        model = SetFitModel.from_pretrained(
            self.model_name,
            labels=self.top_options,
            multi_target_strategy="one-vs-rest",
            use_differentiable_head=True,
            head_params={"out_features": len(self.top_options)},
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
        predict_texts = [self.get_text(sample) for sample in multi_option_data.samples]

        predictions = model.predict(predict_texts)

        return self.predictions_to_options(predictions.tolist())

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

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        return True
