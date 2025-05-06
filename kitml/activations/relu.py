"""
Module contenant l'implémentation de la fonction d'activation ReLU (Rectified Linear Unit).
"""
from kitml.activations.activation import Activation
import numpy as np

class ReLU(Activation):
    """
    Fonction d'activation ReLU (Rectified Linear Unit).
    """
    THRESHOLD = 0.5

    def evaluate(self, x):
        """
        Applique la fonction ReLU à l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            x si x > 0, sinon 0.
        """
        return np.maximum(0, x)

    def derivative(self, x):
        """
        Calcule la dérivée de la fonction ReLU pour l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            1 si x > 0, sinon 0.
        """
        return np.where(x > 0, 1, 0)

