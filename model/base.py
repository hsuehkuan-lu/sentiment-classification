import abc
from torch import nn


class ModelBase(nn.Module):
    @abc.abstractmethod
    def init_weights(self):
        raise NotImplemented

    @abc.abstractmethod
    def forward(self, text, offsets):
        raise NotImplemented

    @abc.abstractmethod
    def load_model(self, model_path):
        raise NotImplemented

    @abc.abstractmethod
    def save_model(self, model_path):
        raise NotImplemented
