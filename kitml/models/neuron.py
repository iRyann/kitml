import numpy as np
import metrics
import activations
from tqdm import tqdm

class Neuron :
    rgn = np.random.default_rng(25)

    # epoch
    def __init__(self, x_train, y_train, x_test, y_test, metric : metrics.Metric, eta, nb_epoch, a : activations.Activation):
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

        c_values = []

        for i in tqdm(range(self.nb_iter)):
            a = self.model()
            
            if i % 100 == 0 :
                c_values.append(self.m.evaluate(self.y_train, a))

            dw, db = self.m.gradientsForNeuron(self.y_train, a)
            self.update(dw, db)
        
        return c_values
        
    def predict(self, x):
        z = self.w.dot(x) + self.b
        return self.activation.evaluate(z) >= 0