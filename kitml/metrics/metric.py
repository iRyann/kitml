"""
Module d'abstraction pour les métriques d'évaluation utilisées dans les modèles de machine learning.
Contient la classe de base abstraite Metric, à hériter pour toute nouvelle métrique.
"""
import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Classe abstraite représentant une métrique d'évaluation.
    Toute métrique personnalisée doit hériter de cette classe et implémenter les méthodes abstraites.
    """
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        Calcule la valeur de la métrique pour les valeurs cibles et les prédictions.
        Args:
            y_true: Valeurs cibles réelles.
            y_pred: Prédictions du modèle.
        Returns:
            Valeur de la métrique.
        """
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """
        Calcule le gradient de la métrique par rapport aux prédictions.
        Args:
            y_true: Valeurs cibles réelles.
            y_pred: Prédictions du modèle.
        Returns:
            Gradient de la métrique.
        """
        pass

    @abstractmethod
    def gradientsForPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron binaire.
        Args:
            y_true: Valeurs cibles réelles.
            y_pred: Prédictions du modèle.
            x: Données d'entrée.
        Returns:
            Tuple contenant les gradients par rapport aux poids et au biais.
        """
        pass

    @abstractmethod
    def gradientsForSingleLayerPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron multi-classes.
        Args:
            y_true: Valeurs cibles réelles.
            y_pred: Prédictions du modèle.
            x: Données d'entrée.
        Returns:
            Tuple contenant les gradients par rapport aux poids et au biais.
        """
        pass


