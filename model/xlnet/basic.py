import torch
import torch.nn as nn
from model.base import ModelBase
from transformers import XLNetModel


class Model(ModelBase):
    def __init__(self, hidden_size, dropout, pretrained_model, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = XLNetModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        x = self.bert(tokens, attention_mask=masks)
        # x = torch.cat(tuple([x.hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        # [CLS] for last 4 layers
        # NOTE: XLNet is in the last index
        x = x.last_hidden_state[:, -1, :]
        x = self.dropout(x)
        return self.out(x).sigmoid()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
