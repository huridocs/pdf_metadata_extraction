import os
import sys

import torch
import pandas as pd
from simpletransformers.t5 import T5Model
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def prepare_dataset(path: str):
    return pd.read_csv(path, sep="\t").astype(str)


def predict(repetition):
    print(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
    test_df = prepare_dataset(f"{SCRIPT_PATH}/data/titles_test.tsv")
    prefix = test_df['prefix'][0]
    model = T5Model("t5", f'outputs/best_model', use_cuda=torch.cuda.is_available())
    with open(f'{SCRIPT_PATH}/results/one_training_{repetition}.txt', 'a') as input_file:
        input_file.write(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
        start = time.time()

        predictions = model.predict([f"{prefix}: {input_text}" for input_text in test_df["input_text"].tolist()])

        end = time.time()
        print('predict time: ', end - start)

        correct = 0

        input_file.write('\nprediction\ntarget\ntext\n\n')

        for prediction, target, text in zip(predictions, test_df["target_text"].tolist(),
                                            test_df["input_text"].tolist()):

            if prediction == target:
                correct += 1
            else:
                text_one_line = text.replace("\n", " ")
                input_file.write(f'{prediction}\n{target}\n{text_one_line}\n')
                input_file.write('\n')

        input_file.write(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
        input_file.write(f'\ntotal:{len(predictions)} errors:{len(predictions) - correct}')
        input_file.write(f'\naccuracy: {100 * correct / len(predictions)}%')
        print(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
        print(f'\ntotal:{len(predictions)} errors:{len(predictions) - correct}')
        print(f'\naccuracy: {100 * correct / len(predictions)}%')

    print(f'\n{time.gmtime().tm_hour}:{time.gmtime().tm_min}\n')
    return 100 * correct / len(predictions)


if __name__ == '__main__':
    predict(1)
