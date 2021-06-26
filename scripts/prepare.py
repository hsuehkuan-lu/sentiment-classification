import os
import json
import torch
from pathlib import Path
from utils import preprocess
from dotenv import load_dotenv

load_dotenv('envs/.env')


if __name__ == '__main__':
    vocab, config = preprocess.generate_vocabulary()
    vocab_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('VOCAB_PATH'))
    torch.save(vocab, vocab_path)
    config_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('CONFIG_PATH'))
    with open(config_path, 'w') as f:
        json.dump(config, f)
