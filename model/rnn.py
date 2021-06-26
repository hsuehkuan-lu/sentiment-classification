import yaml
import torch
from torch import nn
from torch.nn import init
from model.att import Attention
from model.base import ModelBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class LSTMModel(ModelBase):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout, num_classes, attention_method,
                 padding_idx):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, sparse=True, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.attn = Attention(2 * hidden_size, attention_method)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.uniform_(param.data, -initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, text_lengths, hidden=None):
        # text = [L x B]
        sorted_lengths, sorted_idx = text_lengths.sort(descending=True)
        sorted_text = torch.index_select(text, -1, sorted_idx)
        emb = self.embedding(sorted_text)
        packed = nn.utils.rnn.pack_padded_sequence(emb, sorted_lengths.to(torch.device('cpu'), copy=True))
        outputs, hidden = self.lstm(packed, hidden)
        hidden_state, cell_state = hidden
        hidden_state = hidden_state[-2:, :, :].view(1, -1, 2 * self.hidden_size).squeeze(0)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        attn_weights = self.attn(hidden_state, outputs)
        # attn_weights = [batch_size x 1 x lengths]
        context = torch.bmm(attn_weights, outputs.transpose(0, 1)).squeeze(1)
        pred = self.fc(context)
        pred = torch.index_select(pred, 0, torch.arange(0, sorted_idx.shape[0], dtype=torch.int64).to(DEVICE))
        return pred

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
