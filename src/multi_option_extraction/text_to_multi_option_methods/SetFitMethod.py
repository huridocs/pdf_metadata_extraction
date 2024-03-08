import os
import shutil
from os.path import join, exists
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData
from multi_option_extraction.TextToMultiOptionMethod import TextToMultiOptionMethod
from setfit import SetFitModel, SetFitTrainer


class SetFitMethod(TextToMultiOptionMethod):
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

    def get_dataset_from_data(self, multi_option_extraction_data: MultiOptionExtractionData):
        data = list()
        pdf_tags = [x.pdf_tags for x in multi_option_extraction_data.samples]
        texts = [self.get_text_from_pdf_tags(x) for x in pdf_tags]
        labels = self.get_one_hot_encoding(multi_option_extraction_data)

        for text, label in zip(texts, labels):
            data.append([text, label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, multi_option_extraction_data: MultiOptionExtractionData):
        train_dataset = self.get_dataset_from_data(multi_option_extraction_data)
        num_classes = len(self.options)

        model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-mpnet-base-v2",
            use_differentiable_head=True,
            multi_target_strategy="one-vs-rest",
            head_params={"out_features": num_classes},
        )

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=2,
            num_iterations=1,  # The number of text pairs to generate for contrastive learning
            num_epochs=10,  # The number of epochs to use for contrastive learning
        )

        trainer.train(
            output_dir=self.get_model_path(),
            save_strategy="no",
        )

        trainer.train(
            output_dir=self.get_model_path(),
            save_strategy="no",
            load_best_model_at_end=False,
            save_steps=100000,
            eval_steps=100000,
            evaluation_strategy="no",
            save_total_limit=0,
            num_epochs=10,  # The number of epochs to train the head or the whole model (body and head)
            batch_size=2,
            body_learning_rate=1e-5,  # The body's learning rate
            learning_rate=1e-2,  # The head's learning rate
            l2_weight=0.0,  # Weight decay on **both** the body and head. If `None`, will use 0.01.
            max_length=256,
        )

        trainer.model.save_pretrained(self.get_model_path())
        shutil.rmtree("checkpoints", ignore_errors=True)

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[list[Option]]:
        model = SetFitModel.from_pretrained(self.get_model_path())
        predict_texts = [self.get_text_from_pdf_tags(data.pdf_tags) for data in semantic_predictions_data]
        predictions = model(predict_texts)

        return self.predictions_to_options_list(predictions.tolist())
