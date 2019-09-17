import numpy as np


def sin1d(left, right, a, n=100):
    x = np.random.uniform(left, right, (n, 1))
    y = a * np.sin(x)
    return x, y

