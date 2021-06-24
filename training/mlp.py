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
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=PARAMS['train']['optimizer']['lr'])
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS['train']['optimizer']['step_lr'], gamma=PARAMS['train']['optimizer']['gamma']
        )

    @property
    def method(self):
        return 'mlp'
