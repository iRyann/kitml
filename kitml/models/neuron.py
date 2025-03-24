import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Neuron:
    rgn = np.random.default_rng(25)

    def __init__(self, input_size, metric: Metric, eta, nb_epoch, a: Activation):
        self.w = np.reshape(np.array(Neuron.rgn.uniform(size=input_size)), shape=(input_size, 1))
        self.b = Neuron.rgn.uniform()
        self.m = metric
        self.eta = eta
        self.nb_iter = nb_epoch
        self.activation = a

    def model(self, x):
        z = x.dot(self.w) + self.b
        return self.activation.evaluate(z)

    def update(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self, x_train, y_train, threshold=0.5, error_threshold=0.01):
        cost_values = []
        accuracy_values = []

        for i in tqdm(range(self.nb_iter)):
            a = self.model(x_train)

            if i % 10 == 0:
                cost = self.m.evaluate(y_train, a)
                cost_values.append(cost)
                y_pred = self.activation.predict(a, threshold, False) # Already evaluated
                accuracy_values.append(accuracy_score(y_train, y_pred)) # TODO : Standardiser les arrays en entrée

                # Critère d'arrêt basé sur l'erreur
                if cost < error_threshold:
                    print(f"Convergence atteinte à l'itération {i} avec un coût de {cost}.")
                    break

            dw, db = self.m.gradientsForNeuron(y_train, a, x_train)
            self.update(dw, db)

        return cost_values, accuracy_values

    def predict(self, x, threshold=0.5):
        z = x.dot(self.w) + self.b
        return self.activation.predict(z, threshold)
