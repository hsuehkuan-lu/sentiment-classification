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
    def __init__(self, model, train_dataloader, valid_dataloader=None):
        super(MLPTrainer, self).__init__(model, train_dataloader, valid_dataloader)

    @property
    def method(self):
        return 'mlp'
