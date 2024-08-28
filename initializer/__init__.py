import numpy as np
import math
from abc import abstractmethod


class Initializer:
    @abstractmethod
    def initialize(self, *args):
        pass


class ZeroInitializer(Initializer):
    @staticmethod
    def initialize(*args):
        return np.zeros(args)


class XavierInitializer(Initializer):
    @staticmethod
    def initialize(*args):
        return np.random.normal(0, 2 / math.sqrt(sum(args)), args)


class UniformDistribution(Initializer):
    def __init__(self, lower=1, upper=2):
        self.lower = lower
        self.upper = upper

    def initialize(self, *args):
        return np.random.uniform(self.lower, self.upper, args)



class NormalDistribution(Initializer):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def initialize(self, *shape):
        return np.random.normal(self.mean, self.std, shape)
