import yaml
import torch
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class Trainer(TrainerBase):
    def __init__(self, model, dataloader, method, mode):
        super(Trainer, self).__init__(model, dataloader, method, mode)
        self._optimizer = AdamW(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']))
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=int(len(dataloader) * 0.1),
            num_training_steps=len(dataloader) * int(PARAMS[mode]['epochs'])
        )
