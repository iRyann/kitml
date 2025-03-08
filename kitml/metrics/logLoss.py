from metric import Metric
import numpy as np

class LogLoss(Metric) :

    @staticmethod
    def evaluate(y_true, y_pred):

        eps = np.finfo(y_pred.dtype).eps
        p = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    
    @staticmethod
    def gradientsForNeuron(y_true, y_pred, x) :

        dw = 1 / len(y_true) * np.dot(x.T, y_pred - y_true)
        db = 1 / len(y_true) * np.sum(y_pred - y_true)
        return (dw, db)
