from activation import Identity


class Layer:
    def __init__(self, dim, activation=Identity):
        self.dim = dim
        self.activation = activation
