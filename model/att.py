import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    # target is hidden_size
    def __init__(self, hidden_size, method='concat'):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ('dot', 'general', 'concat'):
            raise NotImplemented
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
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

    def dot_score(self, hidden, encoder_output):
        return torch.matmul(hidden, encoder_output)

    def general_score(self, hidden, encoder_output):
        attn = self.attn(encoder_output)
        return torch.matmul(hidden, attn)

    def concat_score(self, hidden, encoder_output):
        hidden_reshape = torch.unsqueeze(hidden, dim=0).repeat(encoder_output.size(0), 1, 1)
        attn = self.attn(torch.cat([hidden_reshape, encoder_output], dim=-1)).tanh()
        return self.v(attn).squeeze(dim=-1)

    def forward(self, hidden, encoder_output):
        # output = [lengths x batch_size x hidden_size]
        # hidden = [batch_size x hidden_size]
        attn_scores = None
        if self.method == 'dot':
            attn_scores = self.dot_score(hidden, encoder_output)
        elif self.method == 'general':
            attn_scores = self.general_score(hidden, encoder_output)
        elif self.method == 'concat':
            attn_scores = self.concat_score(hidden, encoder_output)

        # [lengths x batch_size] -> [batch_size x lengths]
        attn_scores = attn_scores.t()
        # return [batch_size x 1 x lengths]
        return F.softmax(attn_scores, dim=-1).unsqueeze(1)
