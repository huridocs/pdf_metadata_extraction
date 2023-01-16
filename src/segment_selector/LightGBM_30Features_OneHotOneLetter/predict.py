import ast

from typing import Dict
from pathlib import Path
import lightgbm as lgb
from huggingface_hub import hf_hub_download

from config import HUGGINGFACE_PATH
from segment_selector.LightGBM_30Features_OneHotOneLetter.LightGBM_30Features_OneHotOneLetter import (
    LightGBM_30Features_OneHotOneLetter,
)

THIS_SCRIPT_PATH = Path(__file__).parent


def get_model_configs(config_path: str) -> Dict:
    model_configs: {}
    with open(config_path, "r") as config_file:
        config_contents = config_file.read()
        model_configs = ast.literal_eval(config_contents)
    return model_configs


tag_type_finding_config_path = hf_hub_download(
    repo_id="HURIDOCS/pdf-segmetation",
    filename="tag_type_finding_model_config.txt",
    revision="7d98776dd34acb2fe3a06495c82e64b9c84bdc16",
    cache_dir=HUGGINGFACE_PATH,
)

model_configs: {} = get_model_configs(tag_type_finding_config_path)
model_path = hf_hub_download(
    repo_id="HURIDOCS/pdf-segmetation",
    filename="tag_type_finding_model.txt",
    revision="c9e886597823a7995a1454f2de43b821bc930368",
    cache_dir=HUGGINGFACE_PATH,
)

lightgbm_model = lgb.Booster(model_file=model_path)
lightgbm_segmentator = LightGBM_30Features_OneHotOneLetter([], [], model_configs, lightgbm_model)


def set_tag_types(pdf_features):
    tag_types = lightgbm_segmentator.predict(pdf_features)

    for segment in pdf_features.pdf_segments:
        segment.set_type_from_tag_types(tag_types)
