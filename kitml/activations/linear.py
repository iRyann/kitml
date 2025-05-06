"""
Module contenant l'implémentation de la fonction d'activation linéaire.
"""
from kitml.activations.activation import Activation

class Linear(Activation):
    """
    Fonction d'activation linéaire (identité).
    """
    THRESHOLD = 0.5

    def evaluate(self, x):
        """
        Retourne l'entrée sans modification.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La même valeur que x.
        """
        return x

    def derivative(self, x):
        """
        Retourne la dérivée de la fonction linéaire (constante 1).
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            1
        """
        return 1

    def predict(self, z, evaluate=True):
        """
        Retourne la valeur brute z sans classification.
        Args:
            z: Valeur(s) à retourner.
            evaluate: ignoré pour la fonction linéaire.
        Returns:
            z
        """
        return z