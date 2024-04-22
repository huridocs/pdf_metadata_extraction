import os
import shutil
from os.path import join, exists

import pandas as pd
from datasets import load_dataset

from data.ExtractionData import ExtractionData
from data.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from data.PredictionSample import PredictionSample
from extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import EarlyStoppingAfterInitialTraining
from extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod


class TextSetFit(TextToMultiOptionMethod):
    def get_data_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "setfit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    def get_dataset_from_data(self, extraction_data: ExtractionData):
        data = list()
        texts = [self.get_text(sample.tags_texts) for sample in extraction_data.samples]
        labels = self.get_one_hot_encoding(extraction_data)

        for text, label in zip(texts, labels):
            data.append([text, label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, extraction_data: ExtractionData):
        if self.is_multilingual(extraction_data):
            return

        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        train_dataset = self.get_dataset_from_data(extraction_data)
        batch_size = get_batch_size(len(extraction_data.samples))

        model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-mpnet-base-v2",
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

    @staticmethod
    def get_text(texts: list[str]):
        words = list()
        for text in texts:
            text_words = text.split()
            for word in text_words:
                clean_word = "".join([x for x in word if x.isalpha() or x.isdigit()])

                if clean_word:
                    words.append(clean_word)

        return " ".join(words)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        model = SetFitModel.from_pretrained(self.get_model_path())
        texts = [self.get_text(sample.tags_texts) for sample in predictions_samples]
        predictions = model.predict(texts)

        return self.predictions_to_options_list(predictions.tolist())