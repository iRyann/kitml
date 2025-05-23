"""
Module définissant la classe DeepModel pour l'apprentissage profond (réseaux de neurones multicouches).
"""
from kitml.activations.activation import Activation
from kitml.activations.softMax import SoftMax
from kitml.activations.sigmoid import Sigmoid
from kitml.metrics.metric import Metric
from kitml.models.layer import Layer
import numpy as np

class DeepModel:
    """
    Modèle de réseau de neurones profond (multicouches).
    Permet l'entraînement, la prédiction et l'évaluation d'un réseau de couches empilées.
    """
    def __init__(self, layers, learning_rate, loss: Metric, metric: Metric):
        """
        Initialise le modèle profond.
        Args:
            layers: Liste des couches du réseau (instances de Layer).
            learning_rate: Taux d'apprentissage pour la descente de gradient.
            loss: Fonction de coût (métrique de perte).
            metric: Fonction de métrique d'évaluation.
        """
        self.layers = layers
        self.loss = loss
        self.metric = metric
        self.learning_rate = learning_rate
        self.output_size = self.layers[-1].n_out if layers else 0

    def forward(self, x):
        """
        Propage l'entrée x à travers toutes les couches du réseau.
        Args:
            x: Données d'entrée (features, samples).
        Returns:
            Sortie du réseau après la dernière couche.
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def fit(self, x_train, y_train, epochs: int, error_threshold: float, one_hot_encoded=True):
        """
        Entraîne le modèle sur les données d'entraînement.
        Args:
            x_train: Données d'entrée d'entraînement.
            y_train: Labels d'entraînement.
            epochs: Nombre d'époques d'entraînement.
            error_threshold: Seuil d'arrêt sur le coût.
            one_hot_encoded: Indique si y_train est déjà one-hot encodé.
        Returns:
            Tuple (cost_list, metric_list) contenant l'évolution du coût et de la métrique.
        """
        if len(x_train.shape) == 2 and x_train.shape[0] > x_train.shape[1]:
            x_train = x_train.T  # (features, samples)

        # Vérifier si y_train a besoin d'être one-hot encodé
        if isinstance(self.layers[-1].a, SoftMax) and (len(y_train.shape) == 1 or y_train.shape[1] == 1):
            y_train_one_hot = np.zeros((y_train.shape[0], self.output_size))
            for i, y in enumerate(y_train):
                y_train_one_hot[i, int(y)] = 1
            y_train = y_train_one_hot.T  # (feature, samples)
        elif one_hot_encoded and y_train.shape[1] != self.layers[-1].n_out:
            raise ValueError("La dernière couche n'a pas les bonnes dimensions par rapport à y_train")
        else:
            y_train = y_train.T
            
        cost_list = []
        metric_list = []

        for epoch in range(epochs):
            a = self.forward(x_train)
            
            if epoch % 10 == 0 or epoch == (epochs - 1):
                cost = self.loss.evaluate(y_train, a)
                cost_list.append(cost)

                if isinstance(self.layers[-1].a, SoftMax):
                    y_pred = np.argmax(a, axis=0)
                    y_true = np.argmax(y_train, axis=0)
                    accuracy = np.mean(y_pred == y_true)
                    metric_list.append(accuracy)
                    print(f"Époque {epoch}/{epochs}, Coût: {cost:.4f}, Précision: {accuracy:.4f}")
                elif isinstance(self.layers[-1].a, Sigmoid):
                    y_pred = (a > self.layers[-1].a.THRESHOLD).astype(int)
                    accuracy = np.mean(y_pred == y_train)
                    metric_list.append(accuracy)
                    print(f"Époque {epoch}/{epochs}, Coût: {cost:.4f}, Précision: {accuracy:.4f}")
                else:
                    print(f"Époque {epoch}/{epochs}, Coût: {cost:.4f}")

                if cost < error_threshold:
                    print(f"Convergence atteinte à l'itération {epoch} avec un coût de {cost:.4f}.")
                    break
            
            dA = self.loss.gradient(y_train, a)
            for layer in reversed(self.layers):
                dA = layer.backward(dA, self.learning_rate)
                
        return cost_list, metric_list

    def predict(self, x):
        """
        Prédit la sortie du modèle pour de nouvelles données.
        Args:
            x: Données d'entrée à prédire.
        Returns:
            Prédictions du modèle (classe ou valeur continue selon la dernière couche).
        """
        if len(x.shape) == 2 and x.shape[0] > x.shape[1]:
            x = x.T  # (features, samples) si nécessaire
            
        output = self.forward(x)
        
        if isinstance(self.layers[-1].a, SoftMax):
            return np.argmax(output, axis=0)
        elif isinstance(self.layers[-1].a, Sigmoid):
            return (output > self.layers[-1].a.THRESHOLD).astype(int)
        else:
            return output