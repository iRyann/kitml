"""
Module définissant la classe Perceptron pour la classification binaire.
"""
import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from kitml.activations.linear import Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod

class Perceptron:
    """
    Modèle de perceptron pour la classification binaire.
    """
    rgn = np.random.default_rng(25)

    def __init__(self, input_size, metric: Metric, eta, nb_epoch, a: Activation, adaline=False):
        """
        Initialise le perceptron.
        Args:
            input_size: Nombre de caractéristiques en entrée.
            metric: Fonction de coût (métrique de perte).
            eta: Taux d'apprentissage.
            nb_epoch: Nombre d'époques d'entraînement.
            a: Fonction d'activation à utiliser.
            adaline: Booléen indiquant si l'ADALINE est utilisé.
        """
        self.w = np.reshape(np.array(Perceptron.rgn.uniform(size=input_size)), shape=(input_size, 1))
        self.b = Perceptron.rgn.uniform()
        self.m = metric
        self.eta = eta
        self.nb_iter = nb_epoch
        if adaline and not isinstance(a, Linear):
            raise AttributeError("ADALINE activation must be linear")
        self.activation = a
        self.adaline = adaline

    def _y_check_and_reshape(self, y):
        """
        Vérifie et redimensionne le vecteur y.
        Args:
            y: Vecteur des labels.
        Returns:
            Vecteur y redimensionné.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError("L'entrée doit être un vecteur colonne.")
        return y

    def model(self, x):
        """
        Calcule la sortie du modèle pour une entrée donnée.
        Args:
            x: Données d'entrée.
        Returns:
            Sortie du modèle après activation.
        """
        z = x.dot(self.w) + self.b
        return self.activation.evaluate(z)

    def update(self, dw, db):
        """
        Met à jour les poids et le biais.
        Args:
            dw: Gradient des poids.
            db: Gradient du biais.
        """
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self, x_train, y_train, error_threshold=0.01):
        """
        Entraîne le perceptron sur les données d'entraînement.
        Args:
            x_train: Données d'entrée d'entraînement.
            y_train: Labels d'entraînement.
            error_threshold: Seuil d'arrêt sur le coût.
        Returns:
            Tuple contenant les valeurs de coût et d'exactitude à chaque itération.
        """
        y_train = self._y_check_and_reshape(y_train)

        cost_values = []
        accuracy_values = []

        for i in tqdm(range(self.nb_iter)):
            if self.adaline:
                self._train_adaline(x_train, y_train, i, cost_values, accuracy_values, error_threshold)
            else:
                self._train_standard(x_train, y_train, i, cost_values, accuracy_values, error_threshold)

        return cost_values, accuracy_values

    def _train_adaline(self, x_train, y_train, iteration, cost_values, accuracy_values, error_threshold):
        """
        Entraîne le perceptron en mode ADALINE.
        Args:
            x_train: Données d'entrée d'entraînement.
            y_train: Labels d'entraînement.
            iteration: Numéro de l'itération actuelle.
            cost_values: Liste des valeurs de coût.
            accuracy_values: Liste des valeurs d'exactitude.
            error_threshold: Seuil d'arrêt sur le coût.
        """
        for x_i, y_i in zip(x_train, y_train):
            x_i = x_i.reshape(1, -1)
            y_i = y_i.reshape(1, 1)
            a = self.model(x_i)
            dw, db = self.m.gradientsForPerceptron(y_i, a, x_i)
            self.update(dw, db)

        # Évaluation sur l'ensemble des données
        if iteration % 10 == 0:
            a_all = self.model(x_train)
            return self._evaluate_and_check_convergence(y_train, a_all, iteration, cost_values, accuracy_values, error_threshold)

    def _train_standard(self, x_train, y_train, iteration, cost_values, accuracy_values, error_threshold):
        """
        Entraîne le perceptron en mode standard.
        Args:
            x_train: Données d'entrée d'entraînement.
            y_train: Labels d'entraînement.
            iteration: Numéro de l'itération actuelle.
            cost_values: Liste des valeurs de coût.
            accuracy_values: Liste des valeurs d'exactitude.
            error_threshold: Seuil d'arrêt sur le coût.
        """
        a = self.model(x_train)
        dw, db = self.m.gradientsForPerceptron(y_train, a, x_train)
        self.update(dw, db)

        if iteration % 10 == 0:
            end = self._evaluate_and_check_convergence(y_train, a, iteration, cost_values, accuracy_values, error_threshold)
            if(end): return

    def _evaluate_and_check_convergence(self, y_train, a, iteration, cost_values, accuracy_values, error_threshold):
        """
        Évalue le modèle et vérifie la convergence.
        Args:
            y_train: Labels d'entraînement.
            a: Sorties du modèle.
            iteration: Numéro de l'itération actuelle.
            cost_values: Liste des valeurs de coût.
            accuracy_values: Liste des valeurs d'exactitude.
            error_threshold: Seuil d'arrêt sur le coût.
        Returns:
            Booléen indiquant si la convergence a été atteinte.
        """
        cost = self.m.evaluate(y_train, a)
        cost_values.append(cost)
        y_pred = self.activation.predict(a, False)
        # Si on utilise une activation Linear, on est probablement en régression
        if isinstance(self.activation, Linear):
            # Utiliser un coefficient de détermination R² pour la régression
            from sklearn.metrics import r2_score
            performance = r2_score(y_train, y_pred)
            accuracy_values.append(performance)  # On conserve la même liste mais avec une autre métrique
        else:
            accuracy_values.append(accuracy_score(y_train, y_pred))

        if cost < error_threshold:
            print(f"Convergence atteinte à l'itération {iteration} avec un coût de {cost}.")
            return True
        else:
            return False

    def predict(self, x):
        """
        Prédit la classe pour de nouvelles données.
        Args:
            x: Données d'entrée à prédire.
        Returns:
            Prédictions binaires (0 ou 1) ou valeurs continues si activation linéaire.
        """
        z = x.dot(self.w) + self.b
        if isinstance(self.activation, Linear):
            return z
        else:
            return self.activation.predict(z)

