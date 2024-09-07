from abc import abstractmethod, ABC

import numpy as np


class LossFunction:

    @abstractmethod
    def calc(self, target: np.array, pred: np.array):
        pass

    @abstractmethod
    def derivative(self, target: np.array, pred: np.array):
        pass


class L2Norm(LossFunction, ABC):
    def __init__(self, c=0.5):
        self.c = c

    def calc(self, target, pred):
        return self.c * np.sum((target - pred) ** 2) ** 0.5

    def derivative(self, target, pred):
        return self.c * 2 * (target - pred)


class CrossEntropy(LossFunction, ABC):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def calc(self, target, pred):
        # Ensure numerical stability by adding a small value (epsilon) to avoid log(0)
        pred = np.clip(pred, self.epsilon, 1. - self.epsilon)
        return -np.sum(target * np.log(pred))

    def derivative(self, target, pred):
        # Gradient of Cross Entropy Loss
        pred = np.clip(pred, self.epsilon, 1. - self.epsilon)
        return -target / pred
