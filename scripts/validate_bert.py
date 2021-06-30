import os
import sys
import json
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_loader.bert_dataloaders import DataFrameDataLoader
from dotenv import load_dotenv

load_dotenv('envs/.env')


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_validating(method='basic'):
    df = pd.read_csv('data/train.csv')
    train_df, valid_df = train_test_split(df, test_size=1. / PARAMS['validate']['kfold'], random_state=PARAMS['seed'])

    print(f"Train valid split")
    try:
        model_module = importlib.import_module(f'model.bert.{method}')
        model = model_module.Model(**PARAMS['bert'][method])
        model = nn.DataParallel(model)
    except Exception as e:
        raise e
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    try:
        trainer_module = importlib.import_module(f'training.bert.{method}')
        trainer = trainer_module.Trainer(model, mode='validate')
    except Exception as e:
        raise e

    train_dataloader = DataFrameDataLoader(
        train_df, pretrained_model=PARAMS['bert'][method]['pretrained_model'],
        batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], max_len=PARAMS['bert']['max_len']
    )
    valid_dataloader = DataFrameDataLoader(
        valid_df, pretrained_model=PARAMS['bert'][method]['pretrained_model'],
        batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], max_len=PARAMS['bert']['eval_max_len']
    )

    trainer.set_dataloader(train_dataloader, valid_dataloader)

    results, losses = trainer.validate()

    columns = list(losses[0].keys())
    losses_df = pd.DataFrame(losses, columns=columns)

    return results, losses_df


if __name__ == '__main__':
    method = sys.argv[1]
    results, losses_df = start_validating(method)
    results_path = Path(os.getenv('OUTPUT_PATH'), f'bert-{method}_validate_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    plots_path = Path(os.getenv('OUTPUT_PATH'), f'bert-{method}_validate_{os.getenv("PLOTS_PATH")}')
    losses_df.to_csv(plots_path, index=False)