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
from training.bert import Trainer
from dotenv import load_dotenv

load_dotenv('envs/.env')


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_validating(bert_model, pretrained_model, method='basic'):
    df = pd.read_csv('data/train.csv')
    train_df, valid_df = train_test_split(df, test_size=1. / PARAMS['validate']['kfold'], random_state=PARAMS['seed'])

    print(f"Train valid split")
    try:
        model_module = importlib.import_module(f'model.{bert_model}.{method}')
        model = model_module.Model(
            **PARAMS[bert_model], **PARAMS[bert_model][method],
            pretrained_model=pretrained_model
        )
    except Exception as e:
        raise e
    if torch.cuda.is_available():
        device = torch.device('cuda', PARAMS.get('gpu', 0))
    else:
        device = torch.device('cpu')
    model.to(device)

    trainer = Trainer(model, pretrained_model=pretrained_model, mode='validate')

    try:
        dataloader_module = importlib.import_module(f'data_loader.{bert_model}_dataloaders')
    except Exception as e:
        raise e
    train_dataloader = dataloader_module.DataFrameDataLoader(
        train_df, pretrained_model=pretrained_model,
        do_lower_case=PARAMS[bert_model]['do_lower_case'],
        batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], max_len=PARAMS[bert_model]['max_len']
    )
    valid_dataloader = dataloader_module.DataFrameDataLoader(
        valid_df, pretrained_model=pretrained_model,
        do_lower_case=PARAMS[bert_model]['do_lower_case'],
        batch_size=PARAMS['validate']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], max_len=PARAMS[bert_model]['eval_max_len']
    )

    trainer.set_dataloader(train_dataloader, valid_dataloader)
    results, losses = trainer.validate()

    columns = list(losses[0].keys())
    losses_df = pd.DataFrame(losses, columns=columns)

    return results, losses_df


if __name__ == '__main__':
    bert_model, pretrained_model, method = sys.argv[1], sys.argv[2], sys.argv[3]
    results, losses_df = start_validating(bert_model, pretrained_model, method)
    results_path = Path(
        os.getenv('OUTPUT_PATH'),
        f'{bert_model}-{pretrained_model}-{method}_validate_{os.getenv("RESULTS_PATH")}'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f)
    plots_path = Path(
        os.getenv('OUTPUT_PATH'),
        f'{bert_model}-{pretrained_model}-{method}_validate_{os.getenv("PLOTS_PATH")}'
    )
    losses_df.to_csv(plots_path, index=False)
