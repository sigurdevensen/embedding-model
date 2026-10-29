import numpy as np

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> float:
    return sigmoid(x) * (1 - sigmoid(x))