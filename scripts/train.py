import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from data_loader.data_loaders import DataFrameDataLoader
from model import (
    mlp as mlp_model,
    rnn as rnn_model
)
from training import (
    mlp as mlp_trainer,
    rnn as rnn_trainer
)


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


def start_training(method='lstm'):
    if method == 'mlp':
        model = mlp_model.MLPModel(
            CONFIG['vocab_size'], PARAMS[method]['embed_dim'], PARAMS[method]['hidden_size'],
            CONFIG['num_classes'], PARAMS[method]['dropout']
        )
    elif method == 'lstm':
        model = rnn_model.LSTMModel(
            CONFIG['vocab_size'], PARAMS[method]['embed_dim'], PARAMS[method]['hidden_size'],
            PARAMS[method]['n_layers'], PARAMS[method]['dropout'], CONFIG['num_classes'],
            PARAMS[method]['attention_method'], CONFIG['padding_idx']
        )
    else:
        raise NotImplementedError

    kf = KFold(n_splits=PARAMS['train']['kfold'], shuffle=True, random_state=PARAMS['seed'])
    df = pd.read_csv('data/train.csv')
    total_results = list()
    total_losses = list()
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
            trainer = mlp_trainer.MLPTrainer(
                model, train_dataloader, valid_dataloader
            )
        elif method == 'lstm':
            trainer = rnn_trainer.LSTMTrainer(
                model, train_dataloader, valid_dataloader
            )
        else:
            raise NotImplementedError

        results, losses = trainer.train()
        total_results.append(results)
        for loss in losses:
            loss['fold'] = idx + 1
        total_losses += losses
    print(total_losses)
    columns = list(total_losses[0].keys())
    total_losses_df = pd.DataFrame(total_losses, columns=columns)

    average_results = dict()
    for score in ('accuracy', 'precision', 'recall', 'f1-score'):
        average_results[score] = np.mean([results[score] for results in total_results])
    print(average_results)
    return average_results, total_losses_df


if __name__ == '__main__':
    method = sys.argv[1]
    average_results, total_losses_df = start_training(method)
    results_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(average_results, f)
    plots_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_{os.getenv("PLOTS_PATH")}')
    total_losses_df.to_csv(plots_path, index=False)
