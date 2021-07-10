import os
import sys
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('envs/.env')

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def inference(bert_model, pretrained_model, method='lstm'):
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
    model_path = Path(os.getenv('OUTPUT_PATH'), f'{bert_model}-{pretrained_model}-{method}_{os.getenv("MODEL_PATH")}')
    model.load_model(model_path)
    model.to(device)

    df = pd.read_csv('data/test.csv')
    try:
        dataloader_module = importlib.import_module(f'data_loader.{bert_model}_dataloaders')
    except Exception as e:
        raise e
    df[PARAMS['label']] = 0
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = dataloader_module.DataFrameDataLoader(
            df, pretrained_model=pretrained_model,
            do_lower_case=PARAMS[bert_model]['do_lower_case'],
            batch_size=PARAMS['evaluate']['batch_size'], max_len=PARAMS[bert_model]['eval_max_len']
        )
        for idx, (label, text, offsets) in enumerate(tqdm(inference_dataloader)):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.squeeze(dim=-1)
            all_preds += [predicted_label.detach().cpu().numpy()]
    all_preds = np.concatenate(all_preds, axis=0)
    df[PARAMS['label']] = all_preds
    return df


if __name__ == '__main__':
    bert_df = inference('bert', 'bert-large-uncased', 'basic')
    bert_df = bert_df[['ID', PARAMS['label']]]
    roberta_df = inference('roberta', 'roberta-large', 'basic')
    roberta_df = roberta_df[['ID', PARAMS['label']]]
    xlnet_df = inference('xlnet', 'xlnet-large-cased', 'basic')
    xlnet_df = xlnet_df[['ID', PARAMS['label']]]

    df = reduce(
        lambda left, right: pd.merge(left, right, on='ID', how='left'),
        [bert_df, roberta_df, xlnet_df]
    )
    submission_path = Path(os.getenv('OUTPUT_PATH'), 'ensemble.csv')
    df.to_csv(submission_path, index=False)
