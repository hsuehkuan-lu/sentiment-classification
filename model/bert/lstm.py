import torch
import torch.nn as nn
from collections import OrderedDict
from model.base import ModelBase
from model.att import Attention
from transformers import BertModel


class Model(ModelBase):
    def __init__(self, bert_hidden_size, hidden_size, dropout, n_layers, attention_method, pretrained_model,
                 *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            4 * bert_hidden_size, hidden_size, n_layers,
            dropout=dropout, bidirectional=True
        )
        self.attn = Attention(2 * hidden_size, attention_method)
        self.out = nn.Linear(2 * hidden_size, 1)
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        x = self.bert(tokens, attention_mask=masks)
        x = torch.cat(tuple([x.hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        x, (hidden_state, _) = self.lstm(x.transpose(0, 1))
        hidden_state = hidden_state[-2:, :, :].view(1, -1, 2 * self.hidden_size).squeeze(0)
        attn_weights = self.attn(hidden_state, x)
        x = torch.bmm(attn_weights, x.transpose(0, 1)).squeeze(1)
        return self.out(x).sigmoid()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
