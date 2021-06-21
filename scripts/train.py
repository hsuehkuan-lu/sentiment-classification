import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from data_loader.data_loaders import DataFrameDataLoader
from training import mlp, rnn

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_training(method='lstm'):
    kf = KFold(n_splits=PARAMS['train']['kfold'], shuffle=True, random_state=PARAMS['seed'])
    df = pd.read_csv('data/train.csv').iloc[:5]
    total_results = list()
    for idx, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"Cross validation {idx}-fold")
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        train_dataloader = DataFrameDataLoader(
           train_df, batch_size=PARAMS['train']['batch_size'],
           shuffle=PARAMS['train']['shuffle'], use_bag=PARAMS[method]['use_bag']
        )
        valid_dataloader = DataFrameDataLoader(
            valid_df, batch_size=PARAMS['train']['batch_size'],
            shuffle=PARAMS['train']['shuffle'], use_bag=PARAMS[method]['use_bag']
        )

        if method == 'mlp':
            trainer = mlp.MLPTrainer(
                train_dataloader, valid_dataloader
            )
        elif method == 'lstm':
            trainer = rnn.LSTMTrainer(
                train_dataloader, valid_dataloader
            )
        else:
            raise NotImplementedError

        results = trainer.train()
        total_results += [results]
    print(total_results)

    average_results = dict()
    for score in ('accuracy', 'precision', 'recall', 'f1-score'):
        average_results[score] = np.mean([results[score] for results in total_results])
    print(average_results)
    return average_results


if __name__ == '__main__':
    method = sys.argv[1]
    average_results = start_training(method)
    results_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(average_results, f)
