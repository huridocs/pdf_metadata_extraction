import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd
import sentencepiece
import torch
from simpletransformers.config.model_args import T5Args
from simpletransformers.t5 import T5Model

from data.SemanticExtractionData import SemanticExtractionData


class SemanticInformationExtraction:
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, tenant: str, extraction_name: str):
        self.tenant = tenant
        self.extraction_name = extraction_name
        root_folder = f'{Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()}/docker_volume'
        self.semantic_extraction_folder = f'{root_folder}/{self.tenant}/{self.extraction_name}/semantic_model'
        self.model_path = f'{self.semantic_extraction_folder}/best_model'
        self.multilingual_model_path = f'{self.semantic_extraction_folder}/multilingual_best_model'
        self.semantic_extraction_data = list()

    def prepare_dataset(self):
        train_df = pd.DataFrame([[self.extraction_name, x.segment_text, x.text] for x in self.semantic_extraction_data])
        train_df.columns = ['prefix', 'input_text', 'target_text']
        return train_df

    def create_model(self, semantic_extraction_data: List[SemanticExtractionData]):
        self.semantic_extraction_data = semantic_extraction_data
        non_en_extractions = [x for x in semantic_extraction_data if x.language_iso != 'en' and x.language_iso != "eng"]
        multilingual = len(non_en_extractions) > 0
        train_df = self.prepare_dataset()
        model_args = T5Args()
        model_args.max_seq_length = self.get_max_input_length(multilingual)
        model_args.max_length = self.get_max_output_length(multilingual)
        model_args.train_batch_size = 1
        model_args.eval_batch_size = 1
        model_args.num_train_epochs = 3
        model_args.evaluate_during_training = False
        model_args.evaluate_during_training_verbose = False
        model_args.evaluate_during_training_steps = 5000000
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False
        model_args.fp16 = False
        model_args.save_steps = -1
        model_args.use_cached_eval_features = False
        model_args.save_optimizer_and_scheduler = False
        model_args.save_eval_checkpoints = False
        model_args.save_model_every_epoch = False
        model_args.no_cache = True
        model_args.reprocess_input_data = False
        model_args.preprocess_inputs = False
        model_args.num_return_sequences = 1
        model_args.adafactor_eps = (1e-30, 5e-4)
        model_args.early_stopping_consider_epochs = False
        model_args.use_early_stopping = False
        model_args.manual_seed = 42
        model_args.overwrite_output_dir = True
        model_args.tensorboard_dir = f'{self.semantic_extraction_folder}/tensorboard_dir'
        model_args.output_dir = self.multilingual_model_path if multilingual else self.model_path

        print(model_args.output_dir)
        if multilingual:
            model = T5Model("mt5", "google/mt5-small", args=model_args, use_cuda=torch.cuda.is_available())
        else:
            model = T5Model("t5", "t5-small", args=model_args, use_cuda=torch.cuda.is_available())

        model.train_model(train_df)
        print('exists: ', os.path.exists(self.model_path))
        # self.remove_model_if_not_good(train_df)

    def remove_model_if_not_good(self, train_df):
        input_texts = train_df['input_text'].tolist()
        target_texts = train_df['target_text'].tolist()
        predictions = self.get_semantic_predictions(input_texts)

        good_predictions = len([x for index, x in enumerate(predictions) if x == target_texts[index]])
        good_texts_without_t5 = len([x for index, x in enumerate(input_texts) if x == target_texts[index]])
        if good_predictions <= good_texts_without_t5:
            shutil.rmtree(self.model_path, ignore_errors=True)
            shutil.rmtree(self.multilingual_model_path, ignore_errors=True)

    def get_semantic_predictions(self, segments_text: List[str]) -> List[str]:
        if not os.path.exists(self.model_path) and not os.path.exists(self.multilingual_model_path):
            return segments_text

        if os.path.exists(self.model_path):
            model = T5Model("t5", self.model_path, use_cuda=torch.cuda.is_available())
        else:
            model = T5Model("mt5", self.multilingual_model_path, use_cuda=torch.cuda.is_available())

        predictions = model.predict([f"{self.extraction_name}: {input_text}" for input_text in segments_text])
        return predictions

    def get_max_input_length(self, multilingual: bool):
        if multilingual:
            spiece_model = f'{SemanticInformationExtraction.SCRIPT_PATH}/mt5_small_spiece.model'
        else:
            spiece_model = f'{SemanticInformationExtraction.SCRIPT_PATH}/t5_small_spiece.model'

        sentence_piece = sentencepiece.SentencePieceProcessor(spiece_model)
        texts = [self.extraction_name + ': ' + x.segment_text for x in self.semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return int((max(tokens_number) + 1) * 1.2)

    def get_max_output_length(self, multilingual: bool):
        if multilingual:
            spiece_model = f'{SemanticInformationExtraction.SCRIPT_PATH}/mt5_small_spiece.model'
        else:
            spiece_model = f'{SemanticInformationExtraction.SCRIPT_PATH}/t5_small_spiece.model'

        sentence_piece = sentencepiece.SentencePieceProcessor(spiece_model)
        texts = [self.extraction_name + ': ' + x.text for x in self.semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return int((max(tokens_number) + 1) * 1.2)
