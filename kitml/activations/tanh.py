"""
Module contenant l'implémentation de la fonction d'activation tangente hyperbolique (Tanh).
"""
from kitml.activations.activation import Activation
import numpy as np

class Tanh(Activation):
    """
    Fonction d'activation tangente hyperbolique (Tanh).
    """
    THRESHOLD = 0.0

    def evaluate(self, x):
        """
        Applique la fonction tanh à l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La valeur transformée par tanh.
        """
        return np.tanh(x)

    def derivative(self, x):
        """
        Calcule la dérivée de la fonction tanh pour l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La dérivée de tanh appliquée à x.
        """
        e = np.tanh(x)
        return 1 - e ** 2
