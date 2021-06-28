import os
import sys
import json
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from data_loader.data_loaders import DataFrameDataLoader
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


load_dotenv('envs/.env')


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


def start_validating(method='lstm'):
    kf = KFold(n_splits=PARAMS['train']['kfold'], shuffle=True, random_state=PARAMS['seed'])
    df = pd.read_csv('data/train.csv')
    total_results = list()
    total_losses = list()
    dev_loss = None
    for idx, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"Cross validation {idx}-fold")
        try:
            model_module = importlib.import_module(f'model.{method}')
            model = model_module.Model(**CONFIG, **PARAMS[method])
        except Exception as e:
            raise e
        if torch.cuda.is_available():
            device = torch.device('cuda', PARAMS.get('gpu', 0))
        else:
            device = torch.device('cpu')
        model.to(device)

        try:
            trainer_module = importlib.import_module(f'training.{method}')
            trainer = trainer_module.Trainer(model)
        except Exception as e:
            raise e

        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        train_dataloader = DataFrameDataLoader(
            train_df, batch_size=PARAMS['train']['batch_size'],
            shuffle=PARAMS['train']['shuffle'], use_bag=PARAMS[method]['use_bag'],
            use_eos=PARAMS[method].get('use_eos'), max_len=PARAMS[method].get('max_len')
        )
        valid_dataloader = DataFrameDataLoader(
            valid_df, batch_size=PARAMS['train']['batch_size'],
            shuffle=PARAMS['train']['shuffle'], use_bag=PARAMS[method]['use_bag'],
            use_eos=PARAMS[method].get('use_eos'), max_len=PARAMS[method].get('max_len')
        )

        trainer.set_dataloader(train_dataloader, valid_dataloader)

        results, losses, dev_loss = trainer.train(dev_loss)
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
    try:
        average_results, total_losses_df = start_validating(method)
    except Exception as e:
        logging.error(e)
        raise e
    results_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_validate_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(average_results, f)
    plots_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_validate_{os.getenv("PLOTS_PATH")}')
    total_losses_df.to_csv(plots_path, index=False)
