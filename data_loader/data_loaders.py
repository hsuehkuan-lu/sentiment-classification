import yaml
import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.preprocess import Preprocessor

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class DataFrameDataLoader(DataLoader):
    def __init__(self, df, use_bag=True, use_eos=True, max_len=None, *args, **kwargs):
        # order is text, label
        self._preprocessor = None
        self.init()
        self._data_iter = list(zip(df['review'], df['sentiment']))
        collate_batch = partial(self.collate_batch, use_bag=use_bag, use_eos=use_eos, max_len=max_len)
        super(DataFrameDataLoader, self).__init__(self._data_iter, collate_fn=collate_batch, *args, **kwargs)

    def init(self):
        self._preprocessor = Preprocessor(torch.load('outputs/vocab.plk'))

    def collate_batch(self, batch, use_bag, use_eos, max_len):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = self._preprocessor.text_pipeline(_text)
            if use_eos:
                processed_text += [self.vocab[PARAMS['eos_token']]]
            if max_len:
                processed_text = processed_text[:max_len]
            processed_text = torch.tensor(processed_text, dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.float32)
        if use_bag:
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = torch.cat(text_list)
        else:
            offsets = torch.tensor(offsets[1:], dtype=torch.int64)
            text_list = pad_sequence(text_list, padding_value=self.vocab[PARAMS['pad_token']])
        return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)

    @property
    def vocab(self):
        return self._preprocessor.vocab

    @property
    def vocab_size(self):
        return len(self._preprocessor)
