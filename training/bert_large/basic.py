import yaml
import torch
from transformers import AdamW
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class Trainer(TrainerBase):
    def __init__(self, model, mode):
        super(Trainer, self).__init__(model, mode)
        self._optimizer = AdamW(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS[mode]['optimizer']['step_lr'], gamma=PARAMS[mode]['optimizer']['gamma']
        )

    @property
    def method(self):
        return 'bert-large-basic'
