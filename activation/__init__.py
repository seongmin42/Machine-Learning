import math
import numpy as np
from abc import abstractmethod, ABC


class Activation:

    @abstractmethod
    def calc(self, x: np.array):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def __init__(self, a=1, b=0):
        self.a = a
        self.b = b

    def calc(self, x):
        return 1 / (1 + np.exp(-1 * (self.a * x + self.b)))

    def derivative(self, x):
        return self.a * np.where(x < 0, self.calc(x) * (1 - self.calc(x)), (1 - self.calc(-x)) * self.calc(-x))


class Identity(Activation):

    def calc(self, x: np.array):
        return x

    def derivative(self, x: np.array):
        return np.ones(x.shape)


class Relu(Activation):

    def calc(self, x: np.array):
        return x * (x > 0)

    def derivative(self, x: np.array):
        return 1. * (x > 0)


class Softmax(Activation, ABC):

    def calc(self, x: np.array):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x):
        s = self.calc(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)