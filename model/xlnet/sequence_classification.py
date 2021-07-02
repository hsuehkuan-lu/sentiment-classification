import torch
import torch.nn as nn
from model.base import ModelBase
from transformers import XLNetForSequenceClassification


class Model(ModelBase):
    def __init__(self, bert_hidden_size, dropout, pretrained_model, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = XLNetForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
        self.init_weights()

    def forward(self, tokens, masks=None):
        return self.bert(tokens, attention_mask=masks)
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        pass

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
