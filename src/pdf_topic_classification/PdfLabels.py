import pickle
from os.path import join, exists

from config import ROOT_PATH

PARAGRAPHS_CACHE_FOLDER_PATH = join(ROOT_PATH, "data", "paragraphs_cache")


class PdfLabels:
    def __init__(self, pdf_name: str, labels: list[str]):
        self.pdf_name = pdf_name
        self.labels = labels

        paragraph_pickle_path = join(PARAGRAPHS_CACHE_FOLDER_PATH, f"{pdf_name}.pickle")

        if not exists(paragraph_pickle_path):
            return

        with open(paragraph_pickle_path, mode="rb") as file:
            self.paragraphs = pickle.load(file)

    @staticmethod
    def from_dicts(label_dict):
        return PdfLabels(pdf_name=label_dict[0], labels=label_dict[1])
