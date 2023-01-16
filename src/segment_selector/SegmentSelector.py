import shutil
import lightgbm as lgb

from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import List

from config import DATA_PATH
from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from segment_selector.methods.lightgbm_stack_4.LightgbmStack4 import LightgbmStack4
from segment_selector.methods.lightgbm_stack_4_multilingual.LightgbmStack4Multilingual import LightgbmStack4Multilingual


class SegmentSelector:
    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name

        self.base_path = join(DATA_PATH, tenant, property_name)

        self.model_path = join(self.base_path, "segment_predictor_model", "model.model")
        self.multilingual_model_path = join(self.base_path, "multilingual_segment_predictor_model", "model.model")

        self.model, self.multilingual = self.load_model()

    def load_model(self):
        if exists(self.model_path):
            return lgb.Booster(model_file=self.model_path), False

        if exists(self.multilingual_model_path):
            return lgb.Booster(model_file=self.multilingual_model_path), True

        return None, False

    def prepare_model_folder(self):
        shutil.rmtree(Path(self.multilingual_model_path).parent, ignore_errors=True)
        shutil.rmtree(Path(self.model_path).parent, ignore_errors=True)

        model_path = self.multilingual_model_path if self.multilingual else self.model_path

        if not exists(Path(model_path).parent):
            makedirs(Path(model_path).parent)

        return model_path

    def create_model(self, pdfs_features: List[PdfFeatures], multilingual: bool) -> (bool, str):
        self.multilingual = multilingual

        model_path = self.prepare_model_folder()

        self.model = self.get_lightgbm_stack().create_model(pdfs_features)

        if not self.model:
            return False, "No data to create model"

        self.model.save_model(model_path, num_iteration=self.model.best_iteration)

        return True, ""

    def get_lightgbm_stack(self):
        lightgbm_stack = LightgbmStack4Multilingual() if self.multilingual else LightgbmStack4()
        return lightgbm_stack

    def set_extraction_segments(self, pdfs_features: List[PdfFeatures]):
        predictions = self.get_lightgbm_stack().predict(self.model, pdfs_features)
        index = 0
        for pdf_features in pdfs_features:
            for segment in pdf_features.pdf_segments:
                segment.ml_label = 1 if predictions[index] > 0.5 else 0
                index += 1
