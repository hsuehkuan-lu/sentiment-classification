import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from data_loader.data_loaders import DataFrameDataLoader
from training import mlp
from dotenv import load_dotenv

load_dotenv()

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_training():
    kf = KFold(n_splits=PARAMS['train']['kfold'], shuffle=True, random_state=PARAMS['seed'])
    df = pd.read_csv('data/train.csv')
    total_results = list()
    for idx, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"Cross validation {idx}-fold")
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        train_dataloader = DataFrameDataLoader(
           train_df, batch_size=PARAMS['train']['batch_size'],
           shuffle=PARAMS['train']['shuffle']
        )
        valid_dataloader = DataFrameDataLoader(
            valid_df, batch_size=PARAMS['train']['batch_size'],
            shuffle=PARAMS['train']['shuffle']
        )

        if PARAMS['model']['arch'] == 'mlp':
            trainer = mlp.MLPTrainer(
                train_dataloader, valid_dataloader
            )
        else:
            raise NotImplemented

        results = trainer.train()
        total_results += [results]
    print(total_results)

    average_results = dict()
    for score in ('accuracy', 'precision', 'recall', 'f1-score'):
        average_results[score] = np.mean([results[score] for results in total_results])
    print(average_results)
    return average_results


if __name__ == '__main__':
    average_results = start_training()
    with open(os.getenv('RESULTS_PATH'), 'w') as f:
        json.dump(average_results, f)
