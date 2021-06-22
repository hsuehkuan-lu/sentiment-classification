import yaml
import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
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
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = torch.tensor(self._preprocessor.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        if use_bag:
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = torch.cat(text_list)
        else:
            offsets = torch.tensor(offsets[1:], dtype=torch.int64)
            text_list = pad_sequence(text_list, padding_value=self.vocab[PARAMS['pad_token']])
        return label_list.to(self._device), text_list.to(self._device), offsets.to(self._device)

    @property
    def vocab(self):
        return self._preprocessor.vocab

    @property
    def vocab_size(self):
        return len(self._preprocessor)
