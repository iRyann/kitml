import numpy as np
import metrics.metric
from activations import Activation
from tqdm import tqdm

class Neuron :
    rgn = np.random.default_rng(25)

    def __init__(self, x_train, y_train, x_test, y_test, metric, eta, nb_iter, a : Activation):
        self.w = Neuron.rgn.uniform(size = x_train.shape[1])
        self.b = Neuron.rgn.uniform() * np.ones(x_train.shape[1])
        self.m = metric
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.eta = eta
        self.nb_iter = nb_iter
        self.activation = a

    def model(self) :
        z = self.w.dot(self.x_train) + self.b
        return self.activation.evaluate(z)
    
    def cost(self, a):
        return metrics.metric.Metric.log_loss(self.y_train, a)
    
    def gradients(self,a):
        dw = 1 / len(self.y_train) * np.dot(self.x_train.T, a - self.y_train)
        db = 1 / len(self.y_train) * np.sum(a - self.y_train)
        return (dw, db)

    def update(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self):

        c_values = []

        for i in tqdm(range(self.nb_iter)):
            a = self.model()
            
            if i % 100 == 0 :
                c_values.append(self.cost(a))

            dw, db = self.gradients(a)
            self.update(dw, db)
        
        return c_values
        
    def predict(self, x):
        z = self.w.dot(x) + self.b
        return self.activation.evaluate(z) >= 0