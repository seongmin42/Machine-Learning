import numpy as np


class Softmax:

    @staticmethod
    def predict(x: np.array):
        np_sum = np.sum(x)
