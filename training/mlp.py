import yaml
import torch
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class Trainer(TrainerBase):
    def __init__(self, model):
        super(Trainer, self).__init__(model)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=PARAMS['train']['optimizer']['lr'],
                                           weight_decay=float(PARAMS['train']['optimizer']['weight_decay']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS['train']['optimizer']['step_lr'], gamma=PARAMS['train']['optimizer']['gamma']
        )

    @property
    def method(self):
        return 'mlp'
