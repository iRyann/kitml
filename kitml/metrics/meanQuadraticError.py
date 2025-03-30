from kitml.metrics.metric import Metric
import numpy as np

class MeanQuadraticError(Metric):

    def evaluate(self, y_true, y_pred):
        return 0.5 * np.mean(np.square(y_true - y_pred)) 
    
    def gradientsForPerceptron(self, y_true, y_pred, x):
        m = y_true.shape[0]
        dz = y_pred - y_true
        dw = (1 / m) * np.dot(x.T, dz)
        db = (1 / m) * np.sum(dz)
        
        return dw, db
    
    