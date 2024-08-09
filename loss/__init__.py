import numpy as np


class L2Norm:
    def __init__(self, c=0.5):
        self.c = c

    def calc(self, target, pred):
        return self.c * np.sum((target - pred) ** 2) ** 0.5

    def partial_derivative(self, target, pred):
        return self.c * 2 * (target - pred)
