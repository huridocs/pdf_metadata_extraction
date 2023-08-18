import shutil
import lightgbm as lgb

from os import makedirs
from os.path import join, exists
from pathlib import Path

from config import DATA_PATH
from metadata_extraction.PdfSegments import PdfSegments
from segment_selector.methods.lightgbm_frequent_words.LightgbmFrequentWords import LightgbmFrequentWords


class SegmentSelector:
    def __init__(self, tenant: str, extraction_id: str):
        self.tenant = tenant
        self.extraction_id = extraction_id

        self.base_path = join(DATA_PATH, tenant, extraction_id)

        self.model_path = join(self.base_path, "segment_predictor_model", "model.model")

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

    def create_model(self, pdfs_segments: list[PdfSegments]) -> (bool, str):
        model_path = self.prepare_model_folder()

        self.model = LightgbmFrequentWords().create_model(pdfs_segments, model_path)

        if not self.model:
            return False, "No data to create model"

        self.model.save_model(model_path, num_iteration=self.model.best_iteration)

        return True, ""

    def set_extraction_segments(self, pdfs_segments: list[PdfSegments]):
        predictions = LightgbmFrequentWords().predict(self.model, pdfs_segments, self.model_path)
        index = 0
        for pdf_features in pdfs_segments:
            for segment in pdf_features.pdf_segments:
                segment.ml_label = 1 if predictions[index] > 0.5 else 0
                index += 1
