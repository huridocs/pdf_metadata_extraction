import os
import shutil
from os.path import join, exists

import pandas as pd
from datasets import load_dataset

from data.ExtractionData import ExtractionData
from data.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from data.PredictionSample import PredictionSample
from extractors.ExtractorBase import ExtractorBase
from extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import EarlyStoppingAfterInitialTraining
from extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod
from extractors.text_to_multi_option_extractor.methods.TextSetFit import TextSetFit


class TextSetFitMultilingual(TextSetFit):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False
