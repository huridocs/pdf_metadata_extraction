import ast

from typing import Dict
from pathlib import Path
import lightgbm as lgb

from segment_selector.LightGBM_30Features_OneHotOneLetter.LightGBM30FeaturesOneHotOneLetter import (
    LightGBM30FeaturesOneHotOneLetter,
)

THIS_SCRIPT_PATH = Path(__file__).parent


def get_model_configs(config_path: str) -> Dict:
    model_configs: {}
    with open(config_path, "r") as config_file:
        config_contents = config_file.read()
        model_configs = ast.literal_eval(config_contents)
    return model_configs


model_configs: {} = get_model_configs(f"{THIS_SCRIPT_PATH}/config.txt")
lightgbm_model = lgb.Booster(model_file=f"{THIS_SCRIPT_PATH}/model.txt")
lightgbm_segmentator = LightGBM30FeaturesOneHotOneLetter([], [], model_configs, lightgbm_model)


def set_tag_types(pdf_features):
    tag_types = lightgbm_segmentator.predict(pdf_features)

    for segment in pdf_features.pdf_segments:
        segment.set_type_from_tag_types(tag_types)
