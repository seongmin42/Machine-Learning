from model import neural_network as nn
from common.layer import Layer
from initializer import UniformDistribution
import numpy as np

if __name__ == "__main__":
    model = nn.Model()

    model.input(Layer(4))
    model.layer(Layer(3))
    model.layer(Layer(5))
    model.layer(Layer(1, weight_initializer=UniformDistribution(2, 2)))
    model.build()

    X = np.array([[1, 2, 3, 4], [2, 4, 5, 3]])
    y = np.array([[2], [5]])
    model.train(X, y, epochs=1)
