import os
import torch
from pathlib import Path
from utils import preprocess


if __name__ == '__main__':
    vocab = preprocess.generate_vocabulary()
    vocab_path = Path(os.getenv('OUTPUT_PATH'), os.getenv('VOCAB_PATH'))
    torch.save(vocab, vocab_path)
