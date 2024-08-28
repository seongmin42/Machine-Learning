from typing import List

import numpy as np
from common.layer import Layer
from loss import L2Norm
from initializer import NormalDistribution, Initializer, ZeroInitializer


class Model:
    def __init__(self, weight_initializer: Initializer = NormalDistribution(),
                 bias_initializer: Initializer = ZeroInitializer(), loss_function=L2Norm()):
        self.epochs = 10
        self.learning_rate = 0.01
        self.layers: List[Layer] = []

        self.params = {}
        self.weight_initializer: Initializer = weight_initializer
        self.bias_initializer: Initializer = bias_initializer
        self.loss_function = loss_function

        self.compiled = False
        self.losses = []

    def layer(self, layer: Layer):
        self.layers.append(layer)

    # initialize weight and bias
    # W_i is weight between ith and (i+1)th layer
    # b_i is bias which is added to ith layer
    def compile(self):
        if len(self.layers) <= 1:
            raise RuntimeError('Add more layers.')
        for i in range(1, len(self.layers)):
            n, m = self.layers[i - 1].dim, self.layers[i].dim
            self.params[f'W{i}'] = self.weight_initializer.initialize(n, m)
            self.params[f'b{i}'] = self.bias_initializer.initialize(m)

    def train(self, X: np.ndarray, y, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        for e in range(epochs):
            cache = self.feed_forward(X)
            loss = self.evaluate(y, cache[f'H{len(self.layers) - 1}'])
            print(f"epoch:{e+1}, loss: {loss}")
            self.losses.append(loss)
            grads = self.backward_propagation(cache, y)
            self.update_params(grads, self.learning_rate)

    # H_(i+1) = f(U_(i+1)) = f(W_(i+1)·H_i + B_(i+1))
    def feed_forward(self, X: np.ndarray):
        H = X
        cache = {'H0': H}
        for i, layer in enumerate(self.layers[1:]):
            U = np.matmul(H, self.params[f'W{i + 1}']) + self.params[f'b{i + 1}']
            H = layer.activation.calc(U)
            cache[f'U{i + 1}'] = U
            cache[f'H{i + 1}'] = H
        return cache

    def backward_propagation(self, cache, y):
        dE_dH = self.loss_function.partial_derivative(y, cache[f'H{len(self.layers) - 1}'])
        grads = {}
        for i in reversed(range(1, len(self.layers))):
            dH_dU = self.layers[i].activation.derivative(cache[f'U{i}'])
            dE_dU = dE_dH * dH_dU
            grads[f'dW{i}'] = np.matmul(cache[f'H{i - 1}'].T, dE_dU)
            dE_dH = np.matmul(dE_dU, self.params[f'W{i}'].T)
        return grads

    def update_params(self, grads, learning_rate):
        for i in range(1, len(self.layers)):
            self.params[f'W{i}'] += learning_rate * grads[f'dW{i}']

    def evaluate(self, target, y_pred):
        return self.loss_function.calc(target, y_pred)

    def predict(self, X: np.array):
        Y_hat, _ = self.feed_forward(X)
        return np.argmax(Y_hat, axis=1)

    def describe(self):
        print("Neural Network Model".center(50, "="))
        for i, layer in enumerate(self.layers[:-1]):
            print(f"Layer {i}: {layer.dim}")
            print(f"Weight {i + 1}: {self.params[f'W{i + 1}'].shape}")
            print(f"Bias {i + 1}: {self.params[f'b{i + 1}'].shape}")
        print(f"Layer {len(self.layers)}: {self.layers[-1].dim}")
