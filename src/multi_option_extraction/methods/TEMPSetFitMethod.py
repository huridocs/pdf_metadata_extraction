import os
import re
from os.path import join, exists

import pandas as pd
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData
from multi_option_extraction.MultiOptionMethod import MultiOptionMethod
from setfit import SetFitModel, SetFitTrainer


class SetFitMethod(MultiOptionMethod):
    def performance(self, multi_option_extraction_data: MultiOptionExtractionData, training_set_length: int):
        # if not multi_option_extraction_data.samples:
        #     return 0
        #
        # performance_train_set, performance_test_set = self.get_train_test(multi_option_extraction_data, training_set_length)
        #
        # self.train(performance_train_set)
        # prediction_options = self.predict(performance_test_set.to_semantic_prediction_data())
        #
        # self.remove_model()
        # return self.performance_f1_score(performance_test_set, prediction_options)
        pass

    def get_data_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "setfit.model")

    @staticmethod
    def eval_encodings(example):
        example["one_hot_encoding"] = eval(example["one_hot_encoding"])
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

        trainer.freeze()
        trainer.train()

        # Unfreeze the head and freeze the body -> head-only training
        # trainer.unfreeze(keep_body_frozen=True)
        # or
        # Unfreeze the head and unfreeze the body -> end-to-end training
        trainer.unfreeze(keep_body_frozen=False)

        trainer.train(
            num_epochs=10,  # The number of epochs to train the head or the whole model (body and head)
            batch_size=1,
            body_learning_rate=1e-5,  # The body's learning rate
            learning_rate=1e-2,  # The head's learning rate
            l2_weight=0.0,  # Weight decay on **both** the body and head. If `None`, will use 0.01.
            max_length=256,
        )

        trainer.model.save_pretrained(self.get_model_path())

        texts = [self.clean_text(sample.pdf_tags) for sample in multi_option_extraction_data.samples]
        dump(texts, self.get_data_path())

        vectorizer = TfidfVectorizer()
        tfidf_train_vectors = vectorizer.fit_transform(texts)

        labels = self.get_one_hot_encoding(multi_option_extraction_data)
        one_vs_rest_classifier = OneVsRestClassifier(RandomForestClassifier())
        one_vs_rest_classifier = one_vs_rest_classifier.fit(tfidf_train_vectors, labels)
        dump(one_vs_rest_classifier, self.get_model_path())

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[list[Option]]:
        train_texts = load(self.get_data_path())

        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(train_texts)

        predict_texts = [self.clean_text(data.pdf_tags) for data in semantic_predictions_data]

        tfidf_predict_vectors = vectorizer.transform(predict_texts)

        classifier = load(self.get_model_path())
        predictions_text = classifier.predict(tfidf_predict_vectors)
        predictions = [prediction for prediction in predictions_text.tolist()]
        return self.one_hot_to_options_list(predictions)

    def one_hot_to_options_list(self, predictions):
        prediction_options: list[list[Option]] = list()
        for prediction in predictions:
            prediction_options.append(list())
            for i, value in enumerate(prediction):
                if value:
                    prediction_options[-1].append(self.options[i])
        return prediction_options
