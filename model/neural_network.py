from typing import List

import numpy as np
from common.layer import Layer
from loss import L2Norm
from initializer import NormalDistribution, Initializer


class Model:
    def __init__(self, weight_initializer: Initializer=NormalDistribution, bias_initializer: Initializer=NormalDistribution, loss_function=L2Norm):
        self.epochs = 10
        self.learning_rate = 0.01
        self.layers: List[Layer] = []
        self.layer_idx = 0

        self.params = {}
        self.weight_initializer: Initializer = weight_initializer
        self.bias_initializer: Initializer = bias_initializer
        self.loss_function = loss_function

    def input(self, layer: Layer):
        if self.layer_idx > 0:
            raise Exception('Input layer already defined.')
        self.layers.append(layer)

    def layer(self, layer: Layer):
        n, *m = self.layers[self.layer_idx].dim, layer.dim
        self.layers.append(layer)
        self.params[f'W{self.layer_idx}'] = self.weight_initializer.initialize(m)
        self.params[f'b{self.layer_idx}'] = self.bias_initializer.initialize(m)
        self.layer_idx += 1

    def train(self, X: np.ndarray, y, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        for _ in range(epochs):
            pred = self.feed_forward(X)
            self.evaluate(y, pred)
            self.backward_propagation()

    def feed_forward(self, X: np.ndarray):
        H = X
        cache = {'U0': H}
        for i, layer in enumerate(self.layers[1:]):
            U = np.matmul(X, layer.weight_matrix) + layer.bias
            H = layer.activation.calc(U)
            cache[f'U{i+1}'] = U
            cache[f'H{i+1}'] = H
        return cache

    def backward_propagation(self):
        pass

    def evaluate(self, target, y_pred):
        return self.loss_function(target, y_pred)
