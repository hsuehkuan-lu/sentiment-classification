import os
import sys
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from model import mlp, rnn
from data_loader.data_loaders import DataFrameDataLoader

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


def inference(method='lstm'):
    if method == 'mlp':
        model = mlp.MLPModel(
            CONFIG['vocab_size'], PARAMS[method]['embed_dim'], PARAMS[method]['hidden_size'],
            CONFIG['num_classes']
        )
    elif method == 'lstm':
        model = rnn.LSTMModel(
            CONFIG['vocab_size'], PARAMS[method]['embed_dim'], PARAMS[method]['hidden_size'],
            PARAMS[method]['n_layers'], PARAMS[method]['dropout'], CONFIG['num_classes'],
            PARAMS[method]['attention_method'], CONFIG['padding_idx']
        )
    else:
        raise NotImplemented
    model_path = Path(os.getenv('OUTPUT_PATH'), f'{sys.argv[1]}_{os.getenv("MODEL_PATH")}')
    model.load_model(model_path)

    df = pd.read_csv('data/test.csv')
    df[PARAMS['label']] = 0
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = DataFrameDataLoader(df, use_bag=PARAMS[method]['use_bag'])
        for idx, (label, text, offsets) in enumerate(tqdm(inference_dataloader)):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.argmax(1)
            all_preds += [predicted_label.detach().numpy()[0]]
    df[PARAMS['label']] = all_preds
    return df


if __name__ == '__main__':
    method = sys.argv[1]
    df = inference(method)
    df = df[['ID', PARAMS['label']]]
    submission_path = Path(os.getenv('OUTPUT_PATH'), f'{method}_{os.getenv("SUBMISSION_PATH")}')
    df.to_csv(submission_path, index=False)