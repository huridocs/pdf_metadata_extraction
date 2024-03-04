import os
import pickle

from os.path import join, exists
from pathlib import Path

from paragraph_extraction_trainer.Paragraph import Paragraph
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.download_models import paragraph_extraction_model_path
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from config import ROOT_PATH
from pdf_topic_classification.PdfLabels import PARAGRAPHS_CACHE_FOLDER_PATH, PDF_FEATURES_CACHE_FOLDER_PATH
from pdf_topic_classification.pdf_topic_classification_data import get_pdf_names

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")


def cache_paragraph_extraction_predictions():
    for pdf_name in get_pdf_names():
        paragraphs_pickle_path = Path(str(join(PARAGRAPHS_CACHE_FOLDER_PATH, pdf_name + ".pickle")))
        os.makedirs(paragraphs_pickle_path.parent, exist_ok=True)

        if exists(paragraphs_pickle_path):
            continue

        print("caching ", pdf_name)
        pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

        if not pdf_features:
            print(pdf_name)
            continue

        trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
        trainer.set_token_types()
        trainer = ParagraphExtractorTrainer(pdfs_features=[pdf_features], model_configuration=MODEL_CONFIGURATION)
        paragraphs: list[Paragraph] = trainer.get_paragraphs(paragraph_extraction_model_path)

        with open(paragraphs_pickle_path, mode="wb") as file:
            pickle.dump(paragraphs, file)

        with open(paragraphs_pickle_path, mode="wb") as file:
            pickle.dump(paragraphs, file)


def cache_pdf_features():
    for pdf_name in get_pdf_names():
        pdf_features_pickle_path = Path(str(join(PDF_FEATURES_CACHE_FOLDER_PATH, pdf_name + ".pickle")))
        os.makedirs(pdf_features_pickle_path.parent, exist_ok=True)

        if exists(pdf_features_pickle_path):
            continue

        print("caching ", pdf_name)
        pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

        if not pdf_features:
            print(pdf_name)
            continue

        trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
        trainer.set_token_types()

        with open(pdf_features_pickle_path, mode="wb") as file:
            pickle.dump(pdf_features, file)


if __name__ == "__main__":
    cache_paragraph_extraction_predictions()
    cache_pdf_features()
