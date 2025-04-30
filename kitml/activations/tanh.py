from kitml.activations.activation import Activation
import numpy as np

class Tanh(Activation):
    THRESHOLD = 0.0

    def evaluate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        e = np.tanh(x)
        return 1 - e ** 2
