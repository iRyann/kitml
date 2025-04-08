from kitml.metrics.metric import Metric
import numpy as np

class MeanQuadraticError(Metric):
    def evaluate(self, y_true, y_pred):
        """
        Calcule l'erreur quadratique moyenne entre y_true et y_pred.
        
        Args:
            y_true: Les valeurs cibles réelles
            y_pred: Les prédictions du modèle
            
        Returns:
            L'erreur quadratique moyenne * 0.5
        """
        squared_diff = np.square(y_pred - y_true)
        
        mean_squared_error = np.mean(squared_diff)
        
        return 0.5 * mean_squared_error
    
    def gradientsForPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron binaire utilisant MSE.
        
        Args:
            y_true: Les valeurs cibles réelles
            y_pred: Les prédictions du modèle
            x: Les données d'entrée
            
        Returns:
            Tuple contenant les gradients par rapport aux poids et au biais
        """
        m = x.shape[0]  
        dw = 2 / m * np.dot(x.T, (y_pred - y_true))
        db = 2 / m * np.sum(y_pred - y_true)
        
        return (dw, db)
    
    def gradientsForSingleLayerPerceptron(self, y_true, y_pred, x):
        """
        Calcule les gradients pour un perceptron multi-classes utilisant MSE.
        
        Args:
            y_true: Les valeurs cibles réelles de forme (batch_size, output_size)
            y_pred: Les prédictions du modèle de forme (batch_size, output_size)
            x: Les données d'entrée de forme (batch_size, input_size)
            
        Returns:
            Tuple contenant les gradients par rapport aux poids et au biais
        """
        m = x.shape[0]  
        dz = 2 * (y_pred - y_true)  
        dw = 1 / m * np.dot(x.T, dz)
        db = 1 / m * np.sum(dz, axis=0, keepdims=True)  # Forme (1, output_size)
        
        return dw, db

    def gradient(self, y_true, y_pred):

        m = y_true.shape[1] 
        return 2 * (y_pred - y_true) / m
    
    