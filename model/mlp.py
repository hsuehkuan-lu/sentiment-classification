import torch
from torch import nn
from model.base import ModelBase


class MLPModel(ModelBase):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(MLPModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def save_model(self, model_path):
        torch.save(self._model.state_dict(), model_path)
