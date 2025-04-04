from kitml.activations.activation import Activation
import numpy as np

class Sigmoid(Activation) :
    
    THRESHOLD = 0.5

    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        e = self.evaluate(x)
        return e * (1 - e)