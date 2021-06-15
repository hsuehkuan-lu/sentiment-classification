import yaml
import torch
from torch.utils.data import DataLoader
from utils.preprocess import Preprocessor

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class DataFrameDataLoader(DataLoader):
    def __init__(self, df, *args, **kwargs):
        # order is text, label
        self._preprocessor = None
        self._device = None
        self.init()
        self._data_iter = list(zip(df['review'], df['sentiment']))
        super(DataFrameDataLoader, self).__init__(self._data_iter, collate_fn=self.collate_batch, *args, **kwargs)

    def init(self):
        self._preprocessor = Preprocessor(torch.load('outputs/vocab.plk'))
        if torch.cuda.is_available():
            self._device = torch.device('cuda', PARAMS.get('gpu', 0))
        else:
            self._device = torch.device('cpu')

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = torch.tensor(self._preprocessor.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self._device), text_list.to(self._device), offsets.to(self._device)

    @property
    def vocab_size(self):
        return len(self._preprocessor)

    @property
    def num_classes(self):
        return len(set([label for (text, label) in self._data_iter]))
