import os
import sys
import json
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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
    df = pd.read_csv('data/train.csv')
    train_df, valid_df = train_test_split(df, test_size=1. / PARAMS['validate']['kfold'], random_state=PARAMS['seed'])

    print(f"Train valid split")
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
        trainer = trainer_module.Trainer(model, mode='validate')
    except Exception as e:
        raise e

    train_dataloader = DataFrameDataLoader(
        train_df, batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], use_bag=PARAMS[method]['use_bag'],
        use_eos=PARAMS[method].get('use_eos'), max_len=PARAMS[method].get('max_len')
    )
    valid_dataloader = DataFrameDataLoader(
        valid_df, batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], use_bag=PARAMS[method]['use_bag'],
        use_eos=PARAMS[method].get('use_eos'), max_len=PARAMS[method].get('max_len')
    )

    trainer.set_dataloader(train_dataloader, valid_dataloader)
    results, losses = trainer.validate()

    columns = list(losses[0].keys())
    losses_df = pd.DataFrame(losses, columns=columns)

    return results, losses_df


if __name__ == '__main__':
    method = sys.argv[1]
    try:
        results, losses_df = start_validating(method)
    except Exception as e:
        logging.error(e)
        raise e
    results_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_validate_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    plots_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_validate_{os.getenv("PLOTS_PATH")}')
    losses_df.to_csv(plots_path, index=False)
