import os
import sys

import torch
import pandas as pd
from simpletransformers.config.model_args import T5Args
from simpletransformers.t5 import T5Model
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from semantic_information_extraction.sentence_piece import get_max_tokens_number


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def prepare_dataset(path: str):
    return pd.read_csv(path, sep="\t").astype(str)


def get_max_seq_length(prefix, train_df, eval_df):
    max_tokens_input = get_max_tokens_number(
        [f"{prefix}: {input_text}" for input_text in train_df["input_text"].tolist()])
    return max(max_tokens_input,
               get_max_tokens_number([f"{prefix}: {input_text}" for input_text in eval_df["input_text"].tolist()])) + 3


def get_max_length(train_df, eval_df):
    max_tokens_target = get_max_tokens_number(train_df["target_text"].tolist())
    return max(max_tokens_target, get_max_tokens_number(eval_df["target_text"].tolist())) + 3


def train(repetition):
    train_df = prepare_dataset(f"{SCRIPT_PATH}/data/titles_train.tsv")
    train_df = pd.concat([train_df] * 80, ignore_index=True)
    print(len(train_df.index))
    eval_df = train_df.iloc[-10:, :]
    train_df = train_df.iloc[:-10, :]

    print('max_seq_length:', get_max_seq_length(train_df['prefix'][0], train_df, eval_df))
    print('max_length:', get_max_length(train_df, eval_df))

    model_args = T5Args()
    # model_args.max_seq_length = get_max_seq_length(train_df['prefix'][0], train_df, eval_df)
    # model_args.max_length = get_max_length(train_df, eval_df)
    model_args.max_seq_length = 512
    model_args.max_length = 512
    model_args.train_batch_size = 2
    model_args.eval_batch_size = 2
    model_args.num_train_epochs = 2
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = False
    model_args.evaluate_during_training_steps = 5000000
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.use_cached_eval_features = True
    model_args.save_optimizer_and_scheduler = True
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.no_cache = True
    model_args.reprocess_input_data = False
    model_args.preprocess_inputs = False
    model_args.num_return_sequences = 1
    model_args.adafactor_eps = (1e-30, 5e-4)
    model_args.early_stopping_consider_epochs = False
    model_args.use_early_stopping = False
    model_args.manual_seed = repetition
    model_args.overwrite_output_dir = True
    model_args.tensorboard_dir = f'{ROOT_PATH}/files/semantic_models/test'
    model_args.output_dir = f'{ROOT_PATH}/files/semantic_models/test'
    model_args.best_model_dir = f'{ROOT_PATH}/files/semantic_models/test/best_model'

    model = T5Model("mt5", "google/mt5-small", args=model_args, use_cuda=torch.cuda.is_available())

    start = time.time()

    model.train_model(train_df, eval_data=eval_df)

    end = time.time()
    print(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
    print('Train time: ', end - start)


if __name__ == '__main__':
    train(1)
