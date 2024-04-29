import shutil
import lightgbm as lgb

from os import makedirs
from os.path import join, exists
from pathlib import Path

from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfData import PdfData
from extractors.segment_selector.methods.lightgbm_frequent_words.LightgbmFrequentWords import LightgbmFrequentWords


class SegmentSelector:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        self.model_path = join(self.extraction_identifier.get_path(), "segment_predictor_model", "model.model")
        self.model = self.load_model()

    def load_model(self):
        if exists(self.model_path):
            return lgb.Booster(model_file=self.model_path)

        return None

    def prepare_model_folder(self):
        shutil.rmtree(Path(self.model_path).parent, ignore_errors=True)

        model_path = self.model_path

        if not exists(Path(model_path).parent):
            makedirs(Path(model_path).parent)

        return model_path

    def create_model(self, pdfs_data: list[PdfData]) -> (bool, str):
        model_path = self.prepare_model_folder()

        valid_pdf_data = self.get_valid_pdfs_data(pdfs_data)

        if not valid_pdf_data:
            return False, "No data to create model"

        self.model = LightgbmFrequentWords().create_model(valid_pdf_data, model_path)

        if not self.model:
            return False, "No data to create model"

        self.model.save_model(model_path, num_iteration=self.model.best_iteration)

        return True, ""

    @staticmethod
    def get_valid_pdfs_data(pdfs_data: list[PdfData]) -> list[PdfData]:
        valid_pdf_data = list()
        for pdf_data in pdfs_data:
            if not pdf_data.pdf_data_segments or not pdf_data.pdf_features:
                continue
            valid_pdf_data.append(pdf_data)
        return valid_pdf_data

    def set_extraction_segments(self, pdfs_data: list[PdfData]):
        predictions = LightgbmFrequentWords().predict(self.model, pdfs_data, self.model_path)
        index = 0
        for pdf_metadata in pdfs_data:
            for segment in pdf_metadata.pdf_data_segments:
                segment.ml_label = 1 if predictions[index] > 0.5 else 0
                index += 1
