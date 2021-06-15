import time
import yaml
import torch
import pandas as pd
from sklearn.model_selection import KFold
from utils import preprocess
from data_loader.data_loaders import DataFrameDataLoader
from training import mlp

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


def start_training():
    kf = KFold(n_splits=PARAMS['train']['kfold'], shuffle=True, random_state=PARAMS['seed'])
    df = pd.read_csv('data/train.csv')
    total_results = list()
    for idx, (train_index, valid_index) in enumerate(kf.split(df)):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        print(f"Cross validation {idx}-fold")
        train_df = df[train_index]
        valid_df = df[valid_index]

        train_dataloader = DataFrameDataLoader(
           train_df, batch_size=PARAMS['train']['batch_size'],
           shuffle=PARAMS['train']['shuffle']
        )
        valid_dataloader = DataFrameDataLoader(
            valid_df, batch_size=PARAMS['train']['batch_size'],
            shuffle=PARAMS['train']['shuffle']
        )

        if PARAMS['model'] == 'mlp':
            trainer = mlp.MLPTrainer(
                train_dataloader, valid_dataloader
            )
        else:
            raise NotImplemented

        trainer.train()
        total_acc += [cross_acc]
    print(total_acc)
    print(np.mean(total_acc))


if __name__ == '__main__':
    vocab = preprocess.generate_vocabulary()
    torch.save(vocab, 'outputs/vocab.plk')
