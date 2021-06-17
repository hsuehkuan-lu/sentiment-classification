from torch import nn
from torch.nn import init


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)
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
        emb = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(emb, text_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + out
        return self.fc(emb)
