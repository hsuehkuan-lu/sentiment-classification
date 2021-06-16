import os
import json
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from model import mlp
from data_loader.data_loaders import DataFrameDataLoader

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

with open(os.getenv('CONFIG_PATH'), 'r') as f:
    CONFIG = json.load(f)


def inference():
    if PARAMS['model']['arch'] == 'mlp':
        model = mlp.MLPModel(
            CONFIG['vocab_size'], PARAMS['model']['embed_dim'], CONFIG['num_classes']
        )
        model.load_state_dict(torch.load(os.getenv('MODEL_PATH')))
        model.eval()
    else:
        raise NotImplemented

    df = pd.read_csv('data/test.csv')
    df[PARAMS['label']] = 0
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = DataFrameDataLoader(df)
        for idx, (label, text, offsets) in enumerate(tqdm(inference_dataloader)):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.argmax(1)
            all_preds += [predicted_label.detach().numpy()[0]]
    df[PARAMS['label']] = all_preds
    return df


if __name__ == '__main__':
    df = inference()
    df = df[['ID', PARAMS['label']]]
    df.to_csv(os.getenv('SUBMISSION_PATH'))
