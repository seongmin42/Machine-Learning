from loss import CrossEntropy
from model import neural_network as nn
from common.layer import Layer
import numpy as np
from activation import *
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


def get_mnist_data():
    (X_trn, y_trn), (X_tst, y_tst) = mnist.load_data()
    X_trn = X_trn.reshape(X_trn.shape[0], -1) / 255
    y_trn = np.eye(10)[y_trn]
    y_tst = np.eye(10)[y_tst]
    return X_trn, y_trn, X_tst, y_tst, 10


def get_toy_data():
    X_trn = np.arange(40).reshape(10, 4)
    y_trn = np.random.randint(low=0, high=7, size=X_trn.shape[0])
    y_trn = np.eye(7)[y_trn]
    return X_trn, y_trn, 7


def get_training_data(shuffle=False):
    train = np.loadtxt('../Test/training.txt')
    X_trn = train[:, :-1]
    y_trn = train[:, -1]
    y_trn = np.reshape(y_trn, (1000, 1))
    if shuffle:
        indices = np.random.permutation(len(X_trn))
        X_trn = X_trn[indices]
        y_trn = y_trn[indices]
    return X_trn, y_trn, 2


if __name__ == "__main__":
    # X_train, y_train = get_training_data()
    X_train, y_train, X_test, y_test, labels = get_mnist_data()
    # print(X_train.shape, y_train.shape, X_test, y_test)

    model = nn.Model(loss_function=CrossEntropy())

    model.layer(Layer(X_train.shape[1]))
    model.layer(Layer(256, activation=Relu()))
    model.layer(Layer(128, activation=Relu()))
    model.layer(Layer(labels, activation=Sigmoid(a=1e-5)))
    # model.layer(Layer(labels, activation=Relu()))
    model.compile()

    model.train(X_train, y_train, epochs=20, learning_rate=0.001)
    plt.plot(range(20), model.losses)
    plt.show()
