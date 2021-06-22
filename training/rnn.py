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
    def __init__(self, model, train_dataloader, valid_dataloader=None):
        super(LSTMTrainer, self).__init__(model, train_dataloader, valid_dataloader)

    @property
    def method(self):
        return 'lstm'
