"""
Module contenant l'implémentation de la métrique LogLoss (perte logistique/cross-entropy).
"""
from kitml.metrics.metric import Metric
import numpy as np

class LogLoss(Metric):
    """
    Métrique LogLoss pour la classification binaire ou multi-classes.
    """
    def evaluate(self, y_true, y_pred):
        """
        Calcule la perte logistique (cross-entropy) entre y_true et y_pred.
        Args:
            y_true: Vecteur ou matrice des vraies classes (one-hot ou binaire).
            y_pred: Vecteur ou matrice des probabilités prédites.
        Returns:
            Valeur de la perte logistique moyenne.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # y_true est one-hot encoded
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
        return loss

    def gradientsForPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron binaire.
        Args:
            y_true: Vecteur des vraies classes.
            y_pred: Vecteur des probabilités prédites.
            x: Données d'entrée.
        Returns:
            Tuple (dw, db) des gradients.
        """
        m = x.shape[0]
        dw = 1 / m * np.dot(x.T, y_pred - y_true)
        db = 1 / m * np.sum(y_pred - y_true)
        return (dw, db)

    def gradientsForSingleLayerPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron multi-classes.
        Args:
            y_true: Matrice one-hot des vraies classes (output_size, batch_size).
            y_pred: Matrice des probabilités prédites (output_size, batch_size).
            x: Données d'entrée (batch_size, input_size).
        Returns:
            Tuple (dw, db) des gradients.
        """
        m = x.shape[0]
        dz = y_pred - y_true  # Déjà de forme (output_size, batch_size)
        dw = 1 / m * np.dot(dz, x)  # Forme (output_size, input_size)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)  # Forme (output_size, 1)
        return dw, db

    def gradient(self, y_true, y_pred):
        """
        Calcule le gradient de la perte logistique par rapport aux prédictions.
        Args:
            y_true: Vecteur ou matrice des vraies classes.
            y_pred: Vecteur ou matrice des probabilités prédites.
        Returns:
            Différence y_pred - y_true.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return y_pred - y_true
