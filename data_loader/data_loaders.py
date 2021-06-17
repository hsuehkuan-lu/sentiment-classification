import yaml
import torch
import itertools
from functools import partial
from torch.utils.data import DataLoader
from utils.preprocess import Preprocessor

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class DataFrameDataLoader(DataLoader):
    def __init__(self, df, use_bag=True, *args, **kwargs):
        # order is text, label
        self._preprocessor = None
        self._device = None
        self.init()
        self._data_iter = list(zip(df['review'], df['sentiment']))
        collate_batch = partial(self.collate_batch, use_bag=use_bag)
        super(DataFrameDataLoader, self).__init__(self._data_iter, collate_fn=collate_batch, *args, **kwargs)

    def init(self):
        self._preprocessor = Preprocessor(torch.load('outputs/vocab.plk'))
        if torch.cuda.is_available():
            self._device = torch.device('cuda', PARAMS.get('gpu', 0))
        else:
            self._device = torch.device('cpu')

    def collate_batch(self, batch, use_bag):
        label_list, text_list, offsets = [], [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = self._preprocessor.text_pipeline(_text)
            text_list.append(processed_text)
            offsets.append(len(processed_text))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        if use_bag:
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = torch.tensor(list(itertools.chain.from_iterable(text_list)), dtype=torch.int64)
        else:
            offsets = torch.tensor(offsets, dtype=torch.int64)
            text_list = torch.tensor(
                [text[:PARAMS['model']['sent_len']]
                 + [self.vocab[PARAMS['pad_token']]] * (PARAMS['model']['sent_len'] - len(text)) for text in text_list],
                dtype=torch.int64
            )
        return label_list.to(self._device), text_list.to(self._device), offsets.to(self._device)

    @property
    def vocab(self):
        return self._preprocessor.vocab

    @property
    def vocab_size(self):
        return len(self._preprocessor)

    @property
    def num_classes(self):
        return len(set([label for (text, label) in self._data_iter]))
