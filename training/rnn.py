import os
import time
import json
import yaml
import torch
from tqdm import tqdm
from torch.nn import functional as F
from model.rnn import LSTMModel
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class LSTMTrainer(TrainerBase):
    def __init__(self, train_dataloader, valid_dataloader=None):
        super(LSTMTrainer, self).__init__(train_dataloader, valid_dataloader)

    def init_model(self):
        return LSTMModel(
            self.vocab_size, PARAMS['model']['embed_dim'], PARAMS['model']['hidden_size'],
            PARAMS['model']['n_layers'], PARAMS['model']['dropout'], self.num_classes,
            PARAMS['model']['attention_method'], self.padding_idx
        )

    @property
    def method(self):
        return 'lstm'
