"""
Module contenant l'implémentation de la fonction d'activation SoftMax.
"""
import numpy as np
from kitml.activations.activation import Activation

class SoftMax(Activation):
    """
    Fonction d'activation SoftMax pour la classification multi-classes.
    """
    def evaluate(self, z):
        """
        Applique la fonction SoftMax à un vecteur de scores.
        Args:
            z: Tableau numpy de scores (output_size, batch_size) ou (batch_size, output_size).
        Returns:
            Probabilités normalisées pour chaque classe.
        """
        if z.shape[0] <= z.shape[1] or len(z.shape) == 1:
            e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return e_z / np.sum(e_z, axis=0, keepdims=True)
        else:
            e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e_z / np.sum(e_z, axis=1, keepdims=True)

    def derivative(self, x):
        """
        Retourne la dérivée de la fonction SoftMax (ici, valeur indicative 1).
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            1 (la dérivée exacte dépend du contexte d'utilisation).
        """
        return 1
