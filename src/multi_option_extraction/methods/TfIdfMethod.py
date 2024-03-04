import os
import re
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
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData
from multi_option_extraction.MultiOptionMethod import MultiOptionMethod

nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words_set = set(stop_words)

lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)


class TfIdfMethod(MultiOptionMethod):
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

    def clean_text(self, pdf_tags: list[PdfTagData]):
        text = self.get_text_from_pdf_tags(pdf_tags).lower()

        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
        text = re.sub(r"http\S+", "", text)
        html = re.compile(r"<.*?>")
        text = html.sub(r"", text)  # Removing html tags
        punctuations = "@#!?+&*[]-%.:/();$=><|{}^" + "'`" + "_"
        for punctuation in punctuations:
            text = text.replace(punctuation, "")

        text_words = [word for word in text.split() if word not in stop_words_set]
        text = " ".join([lemmatize(word) for word in text_words])

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)  # Removing emojis

        return text

    def train(self, multi_option_extraction_data: MultiOptionExtractionData):
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
        return self.predictions_to_options_list(predictions)
