import time
import yaml
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from model.mlp import MLPModel
from training.basic import TrainerBase

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


class MLPTrainer(TrainerBase):
    def __init__(self, train_dataloader, valid_dataloader=None):
        super(MLPTrainer, self).__init__()
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self.vocab_size = self._train_dataloader.vocab_size
        self.num_classes = self._train_dataloader.num_classes
        self._model = self.init_model()
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=PARAMS['train']['optimizer']['lr'])
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, PARAMS['train']['optimizer']['step_lr'], gamma=PARAMS['train']['optimizer']['gamma']
        )

    def init_model(self):
        return MLPModel(
            self.vocab_size, PARAMS['train']['embed_dim'],
            self.num_classes
        )

    def _train_epoch(self, epoch):
        self._model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predited_label = model(text, offsets)
            loss = criterion(predited_label, F.one_hot(label, num_classes=num_class).type(torch.FloatTensor))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def train(self):
        best_results = dict()
        total_f1 = None
        for epoch in range(1, PARAMS['train']['epochs'] + 1):
            epoch_start_time = time.time()
            self._train_epoch(epoch)
            results = self.evaluate()
            if total_f1 is not None and total_f1 > results['f1-score']:
                self._scheduler.step()
            else:
                total_f1 = results['f1-score']
                best_results = results
            print('-' * 59)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} | precision {:8.3f} | '
                'recall {:8.3f} | f1-score {:8.3f}'.format(
                    epoch, time.time() - epoch_start_time, results['accuracy'],
                    results['precision'], results['recall'], results['f1-score']
                )
            )
            print('-' * 59)
        return best_results

    def evaluate(self):
        self._model.eval()
        total_count = 0
        all_preds, all_labels = list(), list()
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(self._valid_dataloader):
                predicted_label = self._model(text, offsets)
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
            'f1-score': prf[2]
        }
