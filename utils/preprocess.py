import yaml
import torch
from collections import Counter
import pandas as pd
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


tokenizer = get_tokenizer('basic_english')


def generate_vocabulary():
    counter = Counter()
    df = pd.read_csv('data/train.csv')
    for line in df['review']:
        counter.update(tokenizer(line))
    counter = Counter(dict(counter.most_common(PARAMS['basic']['vocab_size'])))
    vocab = Vocab(counter, min_freq=PARAMS['basic']['min_freq'])
    return vocab


class Preprocessor(object):
    def __init__(self, vocab):
        super(Preprocessor, self).__init__()
        self._vocab = vocab
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._vocab)

    def text_pipeline(self, text):
        if isinstance(text, list):
            return [[self._vocab[i] for i in tokenizer(t)] for t in text]
        return [self._vocab[i] for i in tokenizer(text)]

    def label_pipeline(self, label):
        return label
