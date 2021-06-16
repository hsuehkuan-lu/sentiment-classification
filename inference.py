import os
import json
import yaml
import torch
import pandas as pd
from model import mlp
from data_loader.data_loaders import DataFrameDataLoader

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

with open(os.getenv('CONFIG_PATH'), 'r') as f:
    CONFIG = json.load(f)


def inference():
    if PARAMS['model'] == 'mlp':
        model = mlp.MLPModel(
            CONFIG['vocab_size'], PARAMS['model']['embed_dim'], CONFIG['num_classes']
        )
        model.load_state_dict(torch.load(os.getenv('MODEL_PATH')))
        model.eval()
    else:
        raise NotImplemented

    df = pd.read_csv('data/test.csv')
    df[PARAMS['label']] = None
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = DataFrameDataLoader(df)
        for idx, (label, text, offsets) in enumerate(inference_dataloader):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.argmax(1)
            all_preds += [predicted_label.detach().numpy()]
    df[PARAMS['label']] = all_preds
