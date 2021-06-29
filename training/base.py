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
from dotenv import load_dotenv

load_dotenv('envs/.env')

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


class TrainerBase(abc.ABC):
    def __init__(self, model, mode='train'):
        self.mode = mode
        self._train_dataloader = None
        self._valid_dataloader = None
        self._model = model
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS[mode]['optimizer']['step_lr'], gamma=PARAMS[mode]['optimizer']['gamma']
        )
        self._dev_loss = None
        self._early_stops = 0

    def set_dataloader(self, train_dataloader, valid_dataloader=None):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader

    @property
    @abc.abstractmethod
    def method(self):
        raise NotImplementedError

    def save_model(self):
        model_path = Path(os.getenv('OUTPUT_PATH'), f'{self.method}_{os.getenv("MODEL_PATH")}')
        self._model.save_model(model_path)

    def _run_epoch(self, dataloader, is_training=True):
        total_acc, total_count = 0, 0
        all_preds, all_labels = list(), list()
        log_interval = PARAMS['log_interval']
        total_loss = list()
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(tqdm(dataloader)):
            if is_training:
                self._optimizer.zero_grad()
            predicted_label = self._model(text, offsets)
            loss = self._criterion(
                predicted_label,
                F.one_hot(label, num_classes=CONFIG['num_classes']).type(torch.FloatTensor).to(DEVICE)
            )
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), PARAMS[self.mode]['optimizer']['clip'])
                self._optimizer.step()
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    print('| elapsed {} | {:5d}/{:5d} batches | loss {:8.3f} '
                          '| accuracy {:8.3f}'.format(elapsed, idx, len(dataloader), loss, total_acc / total_count))
                    total_acc, total_count = 0, 0
                    start_time = time.time()
            total_loss += [float(loss)]
            predicted_label = predicted_label.argmax(1)
            all_preds += [predicted_label.detach().cpu().numpy()]
            all_labels += [label.detach().cpu().numpy()]
            total_count += label.size(0)
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

    def _train_epoch(self):
        self._model.train()
        return self._run_epoch(self._train_dataloader)

    def validate(self):
        best_results = dict()
        losses = list()
        total_f1 = None
        for epoch in range(1, PARAMS[self.mode]['epochs'] + 1):
            epoch_start_time = time.time()
            train_results = self._train_epoch()
            eval_results = self.evaluate()
            losses.append({
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'dev_loss': eval_results['loss'],
                'learning_rate': self._scheduler.get_last_lr()[0]
            })
            if total_f1 is not None and total_f1 > eval_results['f1-score']:
                self._early_stops += 1
                self._scheduler.step()
            else:
                self._early_stops = 0
                total_f1 = eval_results['f1-score']
                best_results = eval_results
            print('-' * 59)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | avg loss {:8.3f} | '
                'dev loss {:8.3f} | '
                'valid accuracy {:8.3f} | precision {:8.3f} | '
                'recall {:8.3f} | f1-score {:8.3f}'.format(
                    epoch, time.time() - epoch_start_time, train_results['loss'], eval_results['loss'],
                    eval_results['accuracy'], eval_results['precision'], eval_results['recall'],
                    eval_results['f1-score']
                )
            )
            print('-' * 59)
            if self._early_stops == PARAMS[self.mode]['early_stops']:
                break
        return best_results, losses

    def train(self):
        best_results = dict()
        losses = list()
        total_f1 = None
        for epoch in range(1, PARAMS[self.mode]['epochs'] + 1):
            epoch_start_time = time.time()
            results = self._train_epoch()
            losses.append({
                'epoch': epoch,
                'train_loss': results['loss'],
                'dev_loss': 0.,
                'learning_rate': self._scheduler.get_last_lr()[0]
            })
            if total_f1 is not None and total_f1 > results['f1-score']:
                self._early_stops += 1
                if self._early_stops == PARAMS[self.mode]['early_stops']:
                    break
                self._scheduler.step()
            else:
                self._early_stops = 0
                total_f1 = results['f1-score']
                best_results = results
                self.save_model()
            print('-' * 59)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | avg loss {:8.3f} | '
                'valid accuracy {:8.3f} | precision {:8.3f} | '
                'recall {:8.3f} | f1-score {:8.3f}'.format(
                    epoch, time.time() - epoch_start_time, results['loss'], results['accuracy'],
                    results['precision'], results['recall'], results['f1-score']
                )
            )
            print('-' * 59)
        return best_results, losses

    def evaluate(self):
        self._model.eval()

        with torch.no_grad():
            return self._run_epoch(self._valid_dataloader, is_training=False)
