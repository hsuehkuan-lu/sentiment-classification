import yaml
import torch
from torch.utils.data import DataLoader
from utils.preprocess import Preprocessor

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')
PREPROCESSOR = Preprocessor(torch.load('outputs/vocab.plk'))


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(PREPROCESSOR.text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)


class DataFrameDataLoader(DataLoader):
    def __init__(self, df, *args, **kwargs):
        # order is text, label
        data_iter = list(zip(df['review'], df['sentiment']))
        super(DataFrameDataLoader, self).__init__(data_iter, collate_fn=collate_batch, *args, **kwargs)
