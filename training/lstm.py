import yaml
import torch
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class Trainer(TrainerBase):
    def __init__(self, model, method, mode):
        super(Trainer, self).__init__(model, method, mode)
        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']),
                                              weight_decay=float(PARAMS[mode]['optimizer']['weight_decay']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS[mode]['optimizer']['step_lr'], gamma=PARAMS[mode]['optimizer']['gamma']
        )
