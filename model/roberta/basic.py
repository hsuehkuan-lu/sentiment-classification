import torch
import torch.nn as nn
from model.base import ModelBase
from transformers import RobertaModel


class Model(ModelBase):
    def __init__(self, bert_hidden_size, dropout, pretrained_model, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.out = nn.Linear(bert_hidden_size, 1)
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        x = self.bert(tokens, attention_mask=masks)
        x = self.dropout(x.pooler_output)
        x = nn.Tanh()(self.fc(x))
        x = self.dropout(x)
        return nn.Sigmoid()(self.out(x))

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        initializer_range = 0.02
        nn.init.normal_(self.fc.weight, std=initializer_range)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.out.weight, std=initializer_range)
        nn.init.constant_(self.out.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
