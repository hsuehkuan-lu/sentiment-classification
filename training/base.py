import os
import abc
import time
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from dotenv import load_dotenv

load_dotenv('envs/.env')

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', PARAMS.get('gpu', 0))
else:
    DEVICE = torch.device('cpu')


class TrainerBase(abc.ABC):
    def __init__(self, model, dataloader, method, mode='train'):
        self.mode = mode
        self._dataloader = dataloader
        self._model = model
        self._criterion = torch.nn.BCELoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=float(PARAMS[mode]['optimizer']['lr']))
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS[mode]['optimizer']['step_lr'], gamma=PARAMS[mode]['optimizer']['gamma']
        )
        self._dev_loss = None
        self._early_stops = 0
        self.method = method

    def save_model(self):
        model_path = Path(os.getenv('OUTPUT_PATH'), f'{self.method}_{os.getenv("MODEL_PATH")}')
        self._model.save_model(model_path)

    def _run_epoch(self, is_training=True):
        eval_preds, eval_labels = list(), list()
        all_preds, all_labels = list(), list()
        log_interval = PARAMS['log_interval']
        total_loss, eval_loss = list(), list()
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(tqdm(self._dataloader)):
            if is_training:
                self._optimizer.zero_grad()
            predicted_label = self._model(text, offsets)
            loss = self._criterion(
                predicted_label, label.unsqueeze(dim=-1)
            )
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), PARAMS[self.mode]['optimizer']['clip'])
                self._optimizer.step()
                self._scheduler.step()
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    eval_preds = np.concatenate(eval_preds, axis=0)
                    eval_labels = np.concatenate(eval_labels, axis=0)
                    prf = precision_recall_fscore_support(eval_labels, eval_preds, average='binary')
                    print(
                        '| elapsed {} | {:5d}/{:5d} batches | loss {:8.3f} | '
                        'valid accuracy {:8.3f} | precision {:8.3f} | '
                        'recall {:8.3f} | f1-score {:8.3f}'.format(
                            elapsed, idx, len(self._dataloader), np.mean(eval_loss), np.mean(eval_labels == eval_preds),
                            prf[0], prf[1], prf[2]
                        )
                    )
                    eval_preds, eval_labels = list(), list()
                    eval_loss = list()
                    start_time = time.time()
            loss_val = float(loss)
            total_loss += [loss_val]
            eval_loss += [loss_val]
            predicted_label = (predicted_label > 0.5).squeeze(dim=-1)
            p = predicted_label.detach().cpu().numpy()
            l = label.detach().cpu().numpy()
            all_preds += [p]
            all_labels += [l]
            eval_preds += [p]
            eval_labels += [l]
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        prf = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        return {
            'accuracy': np.mean(all_preds == all_labels),
            'precision': prf[0],
            'recall': prf[1],
            'f1-score': prf[2],
            'loss': np.mean(total_loss)
        }

    def _train_epoch(self):
        self._model.train()
        return self._run_epoch()

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
            return self._run_epoch(is_training=False)
