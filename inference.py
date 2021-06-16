import json
import yaml
import torch
from model import mlp

with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)

with open('outputs/config.json', 'r') as f:
    CONFIG = json.load(f)


def inference():
    if PARAMS['model'] == 'mlp':
        model = mlp.MLPModel(
            CONFIG['vocab_size'], PARAMS['model']['embed_dim'], CONFIG['num_classes']
        )
    else:
        raise NotImplemented
