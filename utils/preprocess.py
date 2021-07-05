import yaml
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
    for line in df[PARAMS['feature']]:
        counter.update([''.join(list(filter(lambda x: x.isalpha(), [ch for ch in word]))) for word in tokenizer(line)])
    del counter['']
    num_classes = len(set([label for label in df[PARAMS['label']]]))
    counter = Counter(dict(counter.most_common(PARAMS['basic']['vocab_size'])))
    vocab = Vocab(
        counter, min_freq=PARAMS['basic']['min_freq'],
        specials=(PARAMS['unk_token'], PARAMS['pad_token'], PARAMS['sos_token'], PARAMS['eos_token'])
    )
    config = {
        'vocab_size': len(vocab),
        'num_classes': num_classes,
        'padding_idx': vocab[PARAMS['pad_token']],
        'sos_idx': vocab[PARAMS['sos_token']],
        'eos_idx': vocab[PARAMS['eos_token']]
    }
    return vocab, config


class Preprocessor(object):
    def __init__(self, vocab):
        super(Preprocessor, self).__init__()
        self._vocab = vocab
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._vocab)

    def text_pipeline(self, text):
        if isinstance(text, list):
            return [[self._vocab[''.join(list(filter(lambda x: x.isalpha(), i)))] for i in tokenizer(t)] for t in text]
        return [self._vocab[''.join(list(filter(lambda x: x.isalpha(), i)))] for i in tokenizer(text)]

    def label_pipeline(self, label):
        return label

    @property
    def vocab(self):
        return self._vocab
