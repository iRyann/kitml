from activation import Activation
import numpy as np

class Sigmoid(Activation) :
    
    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))