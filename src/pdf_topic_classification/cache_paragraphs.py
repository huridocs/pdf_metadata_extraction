import json
import os
import pickle
from os import listdir
from os.path import join, exists
from pathlib import Path

from paragraph_extraction_trainer.Paragraph import Paragraph
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.download_models import paragraph_extraction_model_path
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from config import APP_PATH, ROOT_PATH

PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH = join(Path(__file__).parent, 'labeled_data')
PARAGRAPHS_CACHE_FOLDER_PATH = join(APP_PATH, 'data', 'paragraphs_cache')
LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")


def get_pdf_names():
    pdfs_names = set()
    for labeled_data_task in listdir(str(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH)):
        with open(join(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH, labeled_data_task, 'labels.json'), mode="r") as file:
            labels: dict[str, str] | dict[str, list[str]] = json.load(file)
            pdfs_names.update(labels.keys())

    return pdfs_names


def cache_pdfs_features():
    for pdf_name in get_pdf_names():
        pickle_path = Path(join(PARAGRAPHS_CACHE_FOLDER_PATH, pdf_name + ".pickle"))
        os.makedirs(pickle_path.parent, exist_ok=True)

        if exists(pickle_path):
            continue

        print("caching ", pdf_name)
        pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name,  "etree.xml"))
        trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
        trainer.set_token_types()
        trainer = ParagraphExtractorTrainer(pdfs_features=[pdf_features], model_configuration=MODEL_CONFIGURATION)
        paragraphs: list[Paragraph] = trainer.get_paragraphs(paragraph_extraction_model_path)

        with open(pickle_path, mode="wb") as file:
            pickle.dump(paragraphs, file)


if __name__ == '__main__':
    cache_pdfs_features()