import torch
import torch.nn as nn
from collections import OrderedDict
from model.base import ModelBase
from transformers import BertModel


class Model(ModelBase):
    def __init__(self, n_layers, hidden_size, dropout, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        layers = [
            ('fc1', nn.Linear(768, hidden_size)),
            ('drop1', nn.Dropout(dropout)),
            ('relu1', nn.ReLU())
        ]
        for i in range(2, n_layers+1):
            layers += [
                (f'fc{i}', nn.Linear(768))
                (f'drop{i}', nn.Dropout(dropout)),
                (f'relu{i}', nn.ReLU())
            ]
        layers += [
            ('out', nn.Linear(hidden_size, 1)),
            ('sigmoid', nn.Sigmoid())
        ]
        self.out = nn.Sequential(OrderedDict(layers))
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        outputs = self.bert(tokens, attention_mask=masks)
        return self.out(outputs.pooler_output)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        for p in self.bert.parameters():
            p.requires_grad = False
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
