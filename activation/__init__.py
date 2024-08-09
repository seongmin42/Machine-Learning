import math
import numpy as np
from abc import abstractmethod


class Activation:
    @abstractmethod
    def calc(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def __init__(self, a=1, b=0):
        self.a = a
        self.b = b

    def calc(self, x):
        return 1 / (1 + math.exp(-1 * (self.a * x + self.b)))

    def derivative(self, x):
        return self.calc(x) * self.calc(-x)


class Identity(Activation):

    @staticmethod
    def calc(x: np.array):
        return x

    def derivative(self, x: np.array):
        return
