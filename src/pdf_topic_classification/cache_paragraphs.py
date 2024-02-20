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
from pdf_topic_classification.PdfLabels import PARAGRAPHS_CACHE_FOLDER_PATH
from pdf_topic_classification.pdf_topic_classification_data import get_pdf_names

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")


def cache_paragraph_extraction_predictions():
    for pdf_name in get_pdf_names():
        pickle_path = Path(str(join(PARAGRAPHS_CACHE_FOLDER_PATH, pdf_name + ".pickle")))
        os.makedirs(pickle_path.parent, exist_ok=True)

        if exists(pickle_path):
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

        with open(pickle_path, mode="wb") as file:
            pickle.dump(paragraphs, file)


if __name__ == "__main__":
    cache_paragraph_extraction_predictions()
