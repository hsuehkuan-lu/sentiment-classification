import torch
from torch import nn
from model.base import ModelBase


class MLPModel(ModelBase):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout=0.1):
        super(MLPModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.dropout_layer(self.fc1(embedded)).relu()
        x = self.dropout_layer(self.fc2(x)).relu()
        return self.out(x)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
