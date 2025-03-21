import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Neuron :
    rgn = np.random.default_rng(25)

    def __init__(self, x_train, y_train, x_test, y_test, metric : Metric, eta, nb_epoch, a : Activation):
        self.w = Neuron.rgn.uniform(size = x_train.shape[1])
        self.b = Neuron.rgn.uniform() * np.ones(x_train.shape[1])
        self.m = metric
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.eta = eta
        self.nb_iter = nb_epoch
        self.activation = a

    def model(self) :
        z = self.w.dot(self.x_train) + self.b
        return self.activation.evaluate(z)

    def update(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self):

        cost_values = []
        accuracy_values = []

        for i in tqdm(range(self.nb_iter)):
            a = self.model()
            
            if i % 10 == 0 :
                cost_values.append(self.m.evaluate(self.y_train, a))
                accuracy_values.append(accuracy_score(self.y_train, a))

            dw, db = self.m.gradientsForNeuron(self.y_train, a)
            self.update(dw, db)
        
        return cost_values, accuracy_values
        
    def predict(self, x):
        z = self.w.dot(x) + self.b
        return self.activation.evaluate(z) >= 0