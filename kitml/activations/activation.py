"""
Module d'abstraction pour les fonctions d'activation utilisées dans les modèles de machine learning.
Contient la classe de base abstraite Activation, à hériter pour toute nouvelle fonction d'activation.
"""

from abc import abstractmethod, ABC
import numpy as np

class Activation(ABC):
    """
    Classe abstraite représentant une fonction d'activation.
    Toute fonction d'activation personnalisée doit hériter de cette classe et implémenter les méthodes abstraites.
    """
    THRESHOLD = 0.

    @abstractmethod
    def evaluate(self, x):
        """
        Calcule la sortie de la fonction d'activation pour l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La valeur transformée par la fonction d'activation.
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Calcule la dérivée de la fonction d'activation pour l'entrée x.
        Args:
            x: Entrée numérique ou tableau numpy.
        Returns:
            La dérivée de la fonction d'activation appliquée à x.
        """
        pass

    def predict(self, z, evaluate=True):
        """
        Prédit la classe (0 ou 1) à partir de z, en appliquant éventuellement la fonction d'activation.
        Args:
            z: Valeur(s) à classifier.
            evaluate: Si True, applique la fonction d'activation avant la classification.
        Returns:
            0 ou 1 selon le seuil défini.
        """
        res = z
        if evaluate:
            res = self.evaluate(res)
        return np.where(res >= self.THRESHOLD, 1, 0)

