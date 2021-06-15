import abc


class TrainerBase(abc.ABC):
    @abc.abstractmethod
    def train(self):
        raise NotImplemented

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplemented
