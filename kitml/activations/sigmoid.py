"""
Module contenant l'implémentation de la fonction d'activation sigmoïde.
"""
from kitml.activations.activation import Activation
import numpy as np

class Sigmoid(Activation):
    """
    Fonction d'activation sigmoïde.
    """
    THRESHOLD = 0.5

    def evaluate(self, x):
        """
        Applique la fonction sigmoïde à l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La valeur transformée par la sigmoïde.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        Calcule la dérivée de la fonction sigmoïde pour l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La dérivée de la sigmoïde appliquée à x.
        """
        e = self.evaluate(x)
        return e * (1 - e)