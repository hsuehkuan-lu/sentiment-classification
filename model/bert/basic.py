import torch
import torch.nn as nn
from model.base import ModelBase
from transformers import BertModel


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        # [B x L] -> [B x L x D], [B x D]
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(768, 1)
        self.init_weights()

    def forward(self, tokens, masks=None):
        # BERT
        # [B x L x D], [B, D] (pooled_outputs)
        outputs = self.bert(tokens, attention_mask=masks)
        return self.out(outputs.pooler_output).sigmoid()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def init_weights(self):
        for p in self.bert.parameters():
            p.requires_grad = False
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
