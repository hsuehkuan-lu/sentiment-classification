from torch import nn


class Attention(nn.Module):
    # target is hidden_size
    def __init__(self, hidden_size, mode=''):
        super(Attention, self).__init__()
        if mode not in ('dot', 'general', 'concat'):
            raise NotImplemented
        if mode == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif mode == 'concat':
            self.attn = nn.Linear(2 * hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        if hasattr(self, 'attn'):
            self.attn.weight.data.uniform_(-initrange, initrange)
            self.attn.bias.data.zero_()
        if hasattr(self, 'v'):
            self.v.weight.data.uniform_(-initrange, initrange)



    def forward(self, text, text_lengths, hidden=None):
        emb = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(emb, text_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + out
        return self.fc(emb)
