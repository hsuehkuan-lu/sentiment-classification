import os
import sys
import json
import yaml
import torch
import importlib
import pandas as pd
from pathlib import Path
from data_loader.bert_dataloaders import DataFrameDataLoader
from dotenv import load_dotenv


load_dotenv('envs/.env')


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_training(bert_model, method='basic'):
    try:
        model_module = importlib.import_module(f'model.{bert_model}.{method}')
        model = model_module.Model(**PARAMS[bert_model], **PARAMS[bert_model][method])
    except Exception as e:
        raise e
    if torch.cuda.is_available():
        device = torch.device('cuda', PARAMS.get('gpu', 0))
    else:
        device = torch.device('cpu')
    model.to(device)

    try:
        trainer_module = importlib.import_module(f'training.{bert_model}.{method}')
        trainer = trainer_module.Trainer(model, mode='train')
    except Exception as e:
        raise e

    df = pd.read_csv('data/train.csv')
    dataloader = DataFrameDataLoader(
        df, pretrained_model=PARAMS[bert_model]['pretrained_model'],
        batch_size=PARAMS['train']['batch_size'],
        shuffle=PARAMS['validate']['shuffle'], max_len=PARAMS[bert_model]['max_len']
    )
    trainer.set_dataloader(dataloader)
    results, losses = trainer.train()

    columns = list(losses[0].keys())
    losses_df = pd.DataFrame(losses, columns=columns)

    return results, losses_df


if __name__ == '__main__':
    bert_model, method = sys.argv[1], sys.argv[2]
    results, losses_df = start_training(bert_model, method)
    results_path = Path(os.getenv('OUTPUT_PATH'), f'{bert_model}-{method}_{os.getenv("RESULTS_PATH")}')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    plots_path = Path(os.getenv('OUTPUT_PATH'), f'{bert_model}-{method}_{os.getenv("PLOTS_PATH")}')
    losses_df.to_csv(plots_path, index=False)
