import os
import time
import json
import yaml
import torch
from model.mlp import MLPModel
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class MLPTrainer(TrainerBase):
    def __init__(self, train_dataloader, valid_dataloader=None):
        super(MLPTrainer, self).__init__(train_dataloader, valid_dataloader)

    def init_model(self):
        return MLPModel(
            self.vocab_size, PARAMS['model']['embed_dim'],
            self.num_classes
        )
