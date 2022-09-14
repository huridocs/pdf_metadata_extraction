import os
import shutil
from os.path import join, exists
from typing import List

import pandas as pd
from simpletransformers.config.model_args import T5Args
from simpletransformers.t5 import T5Model

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
import sentencepiece
import torch


class T5Method(Method):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    ENGLISH_SENTENCE_PIECE = f"{SCRIPT_PATH}/t5_small_spiece.model"

    def get_model_path(self):
        return join(self.base_path, "t5")

    def get_max_input_length(self, semantic_extraction_data: List[SemanticExtractionData]):
        sentence_piece = sentencepiece.SentencePieceProcessor(self.ENGLISH_SENTENCE_PIECE)

        texts = [self.property_name + ": " + x.segment_text for x in semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return min(int((max(tokens_number) + 1) * 1.2), 250)

    def get_max_output_length(self, semantic_extraction_data: List[SemanticExtractionData]):

        sentence_piece = sentencepiece.SentencePieceProcessor(self.ENGLISH_SENTENCE_PIECE)

        texts = [self.property_name + ": " + x.text for x in semantic_extraction_data]
        tokens_number = [len(sentence_piece.encode(text)) for text in texts]
        return min(int((max(tokens_number) + 1) * 1.2), 250)

    def prepare_dataset(self, semantic_extraction_data: List[SemanticExtractionData]):
        train_df = pd.DataFrame([[self.property_name, x.segment_text, x.text] for x in semantic_extraction_data])
        train_df.columns = ["prefix", "input_text", "target_text"]
        return train_df

    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)
        self.train(performance_train_set)
        predictions = self.predict([x.segment_text for x in performance_test_set])
        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set)

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        self.remove_model_if_exists()
        train_df = self.prepare_dataset(semantic_extraction_data)
        model_args = T5Args()
        model_args.max_seq_length = self.get_max_input_length(semantic_extraction_data)
        model_args.max_length = self.get_max_output_length(semantic_extraction_data)
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
        model_args.tensorboard_dir = f"{self.base_path}/tensorboard_dir"
        model_args.output_dir = self.get_model_path()

        model = T5Model("t5", "t5-small", args=model_args, use_cuda=torch.cuda.is_available())

        model.train_model(train_df)

    def exists_model(self):
        return exists(self.get_model_path())

    def predict(self, texts: List[str]) -> List[str]:
        if not self.exists_model():
            return texts

        model = T5Model("t5", self.get_model_path(), use_cuda=torch.cuda.is_available())
        input_texts = [f"{self.property_name}: {input_text}" for input_text in texts]
        predictions = model.predict(input_texts)
        return predictions

    def remove_model_if_exists(self):
        if self.exists_model():
            shutil.rmtree(self.get_model_path(), ignore_errors=True)
