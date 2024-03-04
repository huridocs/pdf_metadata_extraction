import pickle
from os.path import join

from paragraph_extraction_trainer.Paragraph import Paragraph
from pdf_features.PdfFeatures import PdfFeatures

from config import ROOT_PATH

PARAGRAPHS_CACHE_FOLDER_PATH = join(ROOT_PATH, "data", "paragraphs_cache")
PDF_FEATURES_CACHE_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_features_cache")


class PdfLabels:
    def __init__(self, pdf_name: str, labels: list[str]):
        self.pdf_name = pdf_name
        self.labels = labels

        paragraph_pickle_path = join(PARAGRAPHS_CACHE_FOLDER_PATH, f"{pdf_name}.pickle")
        pdf_features_pickle_path = join(PDF_FEATURES_CACHE_FOLDER_PATH, f"{pdf_name}.pickle")

        with open(paragraph_pickle_path, mode="rb") as file:
            self.paragraphs: list[Paragraph] = pickle.load(file)

        with open(pdf_features_pickle_path, mode="rb") as file:
            self.pdf_features: PdfFeatures = pickle.load(file)

        for paragraph in self.paragraphs:
            paragraph.pdf_name = self.pdf_name

    @staticmethod
    def from_dicts(label_dict):
        return PdfLabels(pdf_name=label_dict[0], labels=label_dict[1])
