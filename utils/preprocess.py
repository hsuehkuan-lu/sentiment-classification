import yaml
import torch
from collections import Counter
import pandas as pd
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def generate_vocabulary():
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    df = pd.read_csv('data/train.csv')
    for line in df['review']:
        counter.update(tokenizer(line))
    counter = Counter(dict(counter.most_common(PARAMS['basic']['vocab_size'])))
    vocab = Vocab(counter, min_freq=PARAMS['basic']['min_freq'])
    return vocab
