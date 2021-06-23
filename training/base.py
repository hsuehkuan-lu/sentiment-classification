import os
import abc
import time
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


class TrainerBase(abc.ABC):
    def __init__(self, model, train_dataloader, valid_dataloader=None):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._model = model
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=PARAMS['train']['optimizer']['lr'])
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS['train']['optimizer']['step_lr'], gamma=PARAMS['train']['optimizer']['gamma']
        )

    @property
    @abc.abstractmethod
    def method(self):
        raise NotImplementedError

    def save_model(self):
        model_path = Path(os.getenv('OUTPUT_PATH'), f'{self.method}_{os.getenv("MODEL_PATH")}')
        self._model.save_model(model_path)

    def _train_epoch(self, epoch):
        self._model.train()
        total_acc, total_count = 0, 0
        log_interval = PARAMS['log_interval']
        start_time = time.time()

        total_loss = list()
        for idx, (label, text, offsets) in enumerate(tqdm(self._train_dataloader)):
            self._optimizer.zero_grad()
            predicted_label = self._model(text, offsets)
            # BCELoss
            loss = self._criterion(
                predicted_label, F.one_hot(label, num_classes=CONFIG['num_classes']).type(torch.FloatTensor)
            )
            loss.backward()
            total_loss += [loss]
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), PARAMS['train']['optimizer']['clip'])
            self._optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | elapsed {} | {:5d}/{:5d} batches | loss {:8.3f} '
                      '| accuracy {:8.3f}'.format(epoch, elapsed, idx, len(self._train_dataloader), loss,
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()
        return torch.mean(torch.stack(total_loss, dim=0)).detach().numpy()

    def train(self):
        best_results = dict()
        best_results['train'] = dict()
        total_f1 = None
        for epoch in range(1, PARAMS['train']['epochs'] + 1):
            epoch_start_time = time.time()
            loss = self._train_epoch(epoch)
            best_results['train'] = {
                'epoch': epoch,
                'loss': loss
            }
            results = self.evaluate()
            if total_f1 is not None and total_f1 > results['f1-score']:
                self._scheduler.step()
            else:
                total_f1 = results['f1-score']
                best_results = results
                self.save_model()
            print('-' * 59)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | avg loss {:8.3f} |'
                'valid accuracy {:8.3f} | precision {:8.3f} | '
                'recall {:8.3f} | f1-score {:8.3f}'.format(
                    epoch, time.time() - epoch_start_time, loss, results['accuracy'],
                    results['precision'], results['recall'], results['f1-score']
                )
            )
            print('-' * 59)
        return best_results

    def evaluate(self):
        self._model.eval()
        total_count = 0
        all_preds, all_labels = list(), list()
        total_loss = list()
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(self._valid_dataloader):
                predicted_label = self._model(text, offsets)
                loss = self._criterion(
                    predicted_label, F.one_hot(label, num_classes=CONFIG['num_classes']).type(torch.FloatTensor)
                )
                total_loss += [loss]
                predicted_label = predicted_label.argmax(1)
                all_preds += [predicted_label.detach().numpy()]
                all_labels += [label.detach().numpy()]
                total_count += label.size(0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        prf = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        return {
            'accuracy': (all_preds == all_labels).mean(),
            'precision': prf[0],
            'recall': prf[1],
            'f1-score': prf[2],
            'loss': torch.mean(torch.stack(total_loss, dim=0)).detach().numpy()
        }
