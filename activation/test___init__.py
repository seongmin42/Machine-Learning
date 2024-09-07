import unittest
import numpy as np
from activation import *
import math


class Test(unittest.TestCase):
    def test_layers(self):
        arr1 = np.arange(12).reshape(4, 3)
        print(f"arr1: {arr1}")
        y = Softmax().calc(arr1)
        print(f"y: {y}")
        d = Softmax().derivative(y)
        print(d)

if __name__ == "__main__":
    unittest.main()
