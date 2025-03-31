import numpy as np
from kitml.activations.activation import Activation

class SoftMax(Activation):
    
    def evaluate(self, z):
        """
        Applique la fonction SoftMax à un vecteur de scores.
        z peut être de forme (output_size, batch_size) ou (batch_size, output_size)
        """
        if z.shape[0] <= z.shape[1] or len(z.shape) == 1:
            e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return e_z / np.sum(e_z, axis=0, keepdims=True)
        else:
            e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e_z / np.sum(e_z, axis=1, keepdims=True)
        
    def derivative(self, x):
        return 1
