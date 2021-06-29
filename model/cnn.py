import yaml
import torch
from collections import OrderedDict
from torch import nn
from model.base import ModelBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class Model(ModelBase):
    def __init__(self, vocab_size, embed_dim, hidden_size, kernel_size, n_layers, dropout, num_classes,
                 padding_idx, *args, **kwargs):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        layers = [
            ('conv1', nn.Conv1d(embed_dim, hidden_size, kernel_size)),
            ('drop1', nn.Dropout(dropout)),
            ('mp1', nn.MaxPool1d(kernel_size)),
            ('relu1', nn.ReLU())
        ]
        for i in range(2, n_layers+1):
            layers += [
                (f'conv{i}', nn.Conv1d(hidden_size, hidden_size, kernel_size)),
                (f'drop{i}', nn.Dropout(dropout)),
                (f'mp{i}', nn.MaxPool1d(kernel_size)),
                (f'relu{i}', nn.ReLU())
            ]
        self.conv = nn.Sequential(OrderedDict(layers))
        self.out = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, text, text_lengths, hidden=None):
        # text = [L x B]
        emb = self.embedding(text)
        # emb = [L x B x D] -> [B x D x L]
        emb = emb.permute(1, 2, 0)
        x = self.conv(emb)
        x, _ = torch.max(x, dim=-1)
        x = self.out(x).sigmoid()
        return x

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
