import time
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AdamW
from training.base import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class Trainer(TrainerBase):
    def __init__(self, model, mode):
        super(Trainer, self).__init__(model, mode)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = AdamW(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS[mode]['optimizer']['step_lr'], gamma=PARAMS[mode]['optimizer']['gamma']
        )

    @property
    def method(self):
        return 'xlnet-sequence-classification'

    def _run_epoch(self, dataloader, is_training=True):
        eval_preds, eval_labels = list(), list()
        all_preds, all_labels = list(), list()
        log_interval = PARAMS['log_interval']
        total_loss = list()
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(tqdm(dataloader)):
            if is_training:
                self._optimizer.zero_grad()
            predicted_label = self._model(text, offsets)
            loss = self._criterion(
                predicted_label, torch.nn.functional.one_hot(label, num_classes=2).to(DEVICE, dtype=torch.int64)
            )
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), PARAMS[self.mode]['optimizer']['clip'])
                self._optimizer.step()
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    eval_preds = np.concatenate(eval_preds, axis=0)
                    eval_labels = np.concatenate(eval_labels, axis=0)
                    prf = precision_recall_fscore_support(eval_labels, eval_preds, average='binary')
                    print(
                        '| elapsed {} | {:5d}/{:5d} batches | loss {:8.3f} | '
                        'valid accuracy {:8.3f} | precision {:8.3f} | '
                        'recall {:8.3f} | f1-score {:8.3f}'.format(
                            elapsed, idx, len(dataloader), loss, (eval_labels == eval_preds).mean(),
                            prf[0], prf[1], prf[2]
                        )
                    )
                    eval_preds, eval_labels = list(), list()
                    start_time = time.time()
            total_loss += [float(loss)]
            preds = torch.argmax(predicted_label, dim=-1)
            p = preds.detach().cpu().numpy()
            l = label.detach().cpu().numpy()
            all_preds += [p]
            all_labels += [l]
            eval_preds += [p]
            eval_labels += [l]
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        prf = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        return {
            'accuracy': (all_preds == all_labels).mean(),
            'precision': prf[0],
            'recall': prf[1],
            'f1-score': prf[2],
            'loss': np.mean(total_loss)
        }
