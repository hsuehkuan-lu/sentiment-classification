import yaml
import torch
from functools import partial
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from utils import preprocess

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class DataFrameDataLoader(DataLoader):
    def __init__(self, df, max_len, pretrained_model, do_lower_case, *args, **kwargs):
        # order is text, label
        self._tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lower_case)
        self._data_iter = list(zip(df['review'], df['sentiment']))
        collate_batch = partial(self.collate_batch, max_len=max_len)
        super(DataFrameDataLoader, self).__init__(self._data_iter, collate_fn=collate_batch, *args, **kwargs)

    def collate_batch(self, batch, max_len):
        label_list, text_list = [], []
        attention_masks = []
        for (_text, _label) in batch:
            label_list.append(_label)
            encoded_dict = self._tokenizer.encode_plus(
                preprocess.preprocess_text(_text, remove_punc=False), add_special_tokens=True, max_length=max_len,
                pad_to_max_length=True, return_attention_mask=True,
                return_tensors='pt'
            )
            text_list.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        label_list = torch.tensor(label_list, dtype=torch.float32)
        text_list = torch.cat(text_list, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return label_list.to(DEVICE), text_list.to(DEVICE), attention_masks.to(DEVICE)
