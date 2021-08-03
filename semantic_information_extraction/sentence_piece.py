import os
from typing import List

import sentencepiece as spm

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_max_tokens_number(texts:List[str]):
    sp = spm.SentencePieceProcessor(model_file=f'{SCRIPT_PATH}/t5_small_spiece.model')
    tokens_number = [len(sp.encode(text)) for text in texts]
    return max(tokens_number)


if __name__ == '__main__':
    text = ''''
32. Having concluded in issues 1 and 2, above, that the plaintiff’s right to his personal liberty has been abused
'''
    sp = spm.SentencePieceProcessor(model_file='mt5_small_spiece.model')

    print(get_max_tokens_number(text))
    print(sp.encode(text, out_type=str))
    print(sp.encode(text))
    print(len(sp.encode(text)))