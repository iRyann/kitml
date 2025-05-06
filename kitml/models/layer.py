"""
Module définissant la classe Layer pour les couches de réseaux de neurones.
"""
import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod

class Layer:
    """
    Représente une couche d'un réseau de neurones.
    """
    rgn = np.random.default_rng(25)

    def __init__(self, n_in: int, n_out: int, activation: Activation):
        """
        Initialise la couche avec le nombre d'entrées, de sorties et la fonction d'activation.
        Args:
            n_in: Nombre d'unités en entrée.
            n_out: Nombre d'unités en sortie.
            activation: Instance de la fonction d'activation à utiliser.
        """
        self.n_in = n_in
        self.n_out = n_out
        
        # Forward
        self.a = activation
        self.w = Layer.rgn.normal(0, np.sqrt(2./n_in), size=(n_out, n_in))  # He
        self.b = np.zeros((n_out, 1))

        # Back-propagation
        self.z = None 
        self.a_in = None
        self.learning_rate = None

    def update(self, dW, db):
        """
        Met à jour les poids et les biais de la couche.
        Args:
            dW: Gradient des poids.
            db: Gradient des biais.
        """
        self.w -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def forward(self, a_in):
        """
        Propage l'entrée à travers la couche.
        Args:
            a_in: Données d'entrée (features, samples).
        Returns:
            Sortie de la couche après activation.
        """
        self.a_in = a_in
        self.z = self.w.dot(a_in) + self.b
        return self.a.evaluate(self.z)
    
    def backward(self, dA, learning_rate):
        """
        Propage le gradient dA en arrière et met à jour les poids et biais.
        Args:
            dA: Gradient du coût par rapport à la sortie de la couche.
            learning_rate: Taux d'apprentissage pour la mise à jour.
        Returns:
            Gradient à propager à la couche précédente.
        """
        self.learning_rate = learning_rate
        m = dA.shape[1] 
        dZ = dA * self.a.derivative(self.z) # On pourrait le pré-calculer dans le cas de sigmoide car grad(s)=s(1-s)
        dW = (1 / m) * dZ.dot(self.a_in.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        self.update(dW,db)

        dA_in = self.w.T.dot(dZ)
        return dA_in



