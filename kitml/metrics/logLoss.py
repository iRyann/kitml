from kitml.metrics.metric import Metric
import numpy as np

class LogLoss(Metric):

    def evaluate(self, y_true, y_pred):
        """
        Calcule la perte logistique (cross-entropy) entre y_true et y_pred.
        y_true et y_pred sont de même forme.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # y_true est one-hot encoded
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
        return loss
    
    def gradientsForPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron binaire.
        """
        m = x.shape[0]
        dw = 1 / m * np.dot(x.T, y_pred - y_true)
        db = 1 / m * np.sum(y_pred - y_true)
        return (dw, db)

    def gradientsForSingleLayerPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron multi-classes.
        y_true: matrice one-hot encoded de forme (output_size, batch_size)
        y_pred: sorties du modèle après softmax de forme (output_size, batch_size)
        x: données d'entrée de forme (batch_size, input_size)
        """
        m = x.shape[0]
        dz = y_pred - y_true  # Déjà de forme (output_size, batch_size)
        dw = 1 / m * np.dot(dz, x)  # Forme (output_size, input_size)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)  # Forme (output_size, 1)
        
        return dw, db
    
    
    def gradient(self, y_true, y_pred):
        
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Pour softmax + entropie croisée, le gradient est simplement (y_pred - y_true)
        return y_pred - y_true
 