from kitml.activations.activation import Activation
import numpy as np

class ReLU(Activation) :
    
    THRESHOLD = 0.5

    def evaluate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
