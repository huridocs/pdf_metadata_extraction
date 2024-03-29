import os
from functools import lru_cache
from os.path import join, exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from data.Option import Option
from multi_option_extraction.MultiLabelMethod import MultiLabelMethod
from multi_option_extraction.data.MultiOptionData import MultiOptionData

nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words_set = set(stop_words)

lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)


class TfIdfMethod(MultiLabelMethod):
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

    def train(self, multi_option_data: MultiOptionData):
        texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]
        dump(texts, self.get_data_path())

        vectorized = TfidfVectorizer()
        tfidf_train_vectors = vectorized.fit_transform(texts)

        labels = self.get_one_hot_encoding(multi_option_data)
        one_vs_rest_classifier = OneVsRestClassifier(RandomForestClassifier())
        one_vs_rest_classifier = one_vs_rest_classifier.fit(tfidf_train_vectors, labels)
        dump(one_vs_rest_classifier, self.get_model_path())

    def predict(self, multi_option_data: MultiOptionData) -> list[list[Option]]:
        train_texts = load(self.get_data_path())

        vectorized = TfidfVectorizer()
        vectorized.fit_transform(train_texts)

        predict_texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]

        tfidf_predict_vectors = vectorized.transform(predict_texts)

        classifier = load(self.get_model_path())
        predictions_text = classifier.predict(tfidf_predict_vectors)
        predictions = [prediction for prediction in predictions_text.tolist()]
        return self.predictions_to_options_list(predictions)
