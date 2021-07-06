import os
import sys
import json
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from data_loader.data_loaders import DataFrameDataLoader
from dotenv import load_dotenv

load_dotenv('envs/.env')

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


def inference(method='lstm'):
    try:
        model_module = importlib.import_module(f'model.{method}')
        model = model_module.Model(**CONFIG, **PARAMS[method])
    except Exception as e:
        raise e
    model_path = Path(os.getenv('OUTPUT_PATH'), f'{sys.argv[1]}_{os.getenv("MODEL_PATH")}')
    model.load_model(model_path)
    if torch.cuda.is_available():
        device = torch.device('cuda', PARAMS.get('gpu', 0))
    else:
        device = torch.device('cpu')
    model.to(device)

    df = pd.read_csv('data/test.csv')
    df[PARAMS['label']] = 0
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = DataFrameDataLoader(
            df, batch_size=PARAMS['evaluate']['batch_size'], use_bag=PARAMS[method]['use_bag'],
            use_eos=PARAMS[method].get('use_eos'), max_len=PARAMS[method].get('max_len')
        )
        for idx, (label, text, offsets) in enumerate(tqdm(inference_dataloader)):
            predicted_label = model(text, offsets)
            predicted_label = (predicted_label > 0.5).squeeze(dim=-1)
            all_preds += [predicted_label.detach().cpu().numpy().astype(int)]
    all_preds = np.concatenate(all_preds, axis=0)
    df[PARAMS['label']] = all_preds
    return df


if __name__ == '__main__':
    method = sys.argv[1]
    df = inference(method)
    df = df[['ID', PARAMS['label']]]
    submission_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_{os.getenv("SUBMISSION_PATH")}')
    df.to_csv(submission_path, index=False)
