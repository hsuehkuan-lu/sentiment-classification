import torch
from utils import preprocess


if __name__ == '__main__':
    vocab = preprocess.generate_vocabulary()
    torch.save(vocab, 'outputs/vocab.plk')
