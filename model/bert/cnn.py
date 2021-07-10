import torch
import torch.nn as nn
from model.base import ModelBase
from transformers import BertModel


class Model(ModelBase):
    def __init__(self, bert_hidden_size, hidden_size, kernel_size, dropout, pretrained_model, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(bert_hidden_size, hidden_size, kernel_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        x = self.bert(tokens, attention_mask=masks)
        x = self.dropout(x.last_hidden_state)
        x = self.conv(x.permute(0, 2, 1))
        x = torch.max(x, 2)[0]
        return nn.Sigmoid()(self.out(x))

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        def init_conv(m):
            if type(m) == nn.Conv1d:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        initializer_range = 0.02
        self.conv.apply(init_conv)
        nn.init.normal_(self.out.weight, std=initializer_range)
        nn.init.constant_(self.out.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
