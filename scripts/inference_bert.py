import os
import sys
import yaml
import torch
import importlib
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from data_loader.bert_dataloaders import DataFrameDataLoader
from dotenv import load_dotenv

load_dotenv('envs/.env')

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def inference(bert_model, method='lstm'):
    try:
        model_module = importlib.import_module(f'model.{bert_model}.{method}')
        model = model_module.Model(**PARAMS[bert_model], **PARAMS[bert_model][method])
    except Exception as e:
        raise e
    if torch.cuda.is_available():
        device = torch.device('cuda', PARAMS.get('gpu', 0))
    else:
        device = torch.device('cpu')
    model_path = Path(os.getenv('OUTPUT_PATH'), f'{bert_model}-{method}_{os.getenv("MODEL_PATH")}')
    model.load_model(model_path)
    model.to(device)

    df = pd.read_csv('data/test.csv')
    df[PARAMS['label']] = 0
    with torch.no_grad():
        all_preds = list()
        inference_dataloader = DataFrameDataLoader(
            df, pretrained_model=PARAMS[bert_model]['pretrained_model'],
            batch_size=PARAMS['evaluate']['batch_size'], max_len=PARAMS[bert_model]['eval_max_len']
        )
        for idx, (label, text, offsets) in enumerate(tqdm(inference_dataloader)):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.argmax(1)
            all_preds += predicted_label.detach().cpu().numpy().tolist()
    df[PARAMS['label']] = all_preds
    return df


if __name__ == '__main__':
    bert_model, method = sys.argv[1], sys.argv[2]
    df = inference(bert_model, method)
    df = df[['ID', PARAMS['label']]]
    submission_path = Path(os.getenv('OUTPUT_PATH'), f'{bert_model}-{method}_{os.getenv("SUBMISSION_PATH")}')
    df.to_csv(submission_path, index=False)
