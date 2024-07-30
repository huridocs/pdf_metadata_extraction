import os
from functools import lru_cache
from os.path import join, exists
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ROOT_PATH
from data.Option import Option
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from data.ExtractionData import ExtractionData

nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words_set = set(stop_words)

lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)


class TfIdfMethod(MultiLabelMethod):

    @staticmethod
    def get_text(sample: TrainingSample) -> str:
        file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
        text = Path(ROOT_PATH, "data", "cyrilla_summaries", file_name).read_text()

        if "three sentence" in text.split(":")[0]:
            text = ":".join(text.split(":")[1:]).strip()

        return text if text else "No text"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_data_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.txt")

    def get_model_path(self):
        model_folder_path = join(self.base_path, self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "fast.model")

    def train(self, multi_option_data: ExtractionData):
        texts = [self.get_text(sample) for sample in multi_option_data.samples]
        dump(texts, self.get_data_path())

        vectorized = TfidfVectorizer()
        tfidf_train_vectors = vectorized.fit_transform(texts)

        labels = self.get_one_hot_encoding(multi_option_data)
        one_vs_rest_classifier = OneVsRestClassifier(RandomForestClassifier())
        one_vs_rest_classifier = one_vs_rest_classifier.fit(tfidf_train_vectors, labels)
        dump(one_vs_rest_classifier, self.get_model_path())

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        train_texts = load(self.get_data_path())

        vectorized = TfidfVectorizer()
        vectorized.fit_transform(train_texts)

        predict_texts = [self.get_text(sample) for sample in multi_option_data.samples]

        tfidf_predict_vectors = vectorized.transform(predict_texts)

        classifier = load(self.get_model_path())
        predictions_text = classifier.predict(tfidf_predict_vectors)
        predictions = [prediction for prediction in predictions_text.tolist()]
        return self.predictions_to_options_list(predictions)
