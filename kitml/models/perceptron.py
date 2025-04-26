import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from kitml.activations.linear import Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod

class Perceptron:
    rgn = np.random.default_rng(25)

    def __init__(self, input_size, metric: Metric, eta, nb_epoch, a: Activation, adaline=False):
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
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError("L'entrée doit être un vecteur colonne.")
        return y

    def model(self, x):
        z = x.dot(self.w) + self.b
        return self.activation.evaluate(z)

    def update(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self, x_train, y_train, error_threshold=0.01):
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
        a = self.model(x_train)
        dw, db = self.m.gradientsForPerceptron(y_train, a, x_train)
        self.update(dw, db)

        if iteration % 10 == 0:
            end = self._evaluate_and_check_convergence(y_train, a, iteration, cost_values, accuracy_values, error_threshold)
            if(end) : return
            
    def _evaluate_and_check_convergence(self, y_train, a, iteration, cost_values, accuracy_values, error_threshold):
        cost = self.m.evaluate(y_train, a)
        cost_values.append(cost)
        y_pred = self.activation.predict(a, False)
        accuracy_values.append(accuracy_score(y_train, y_pred))

        if cost < error_threshold:
            print(f"Convergence atteinte à l'itération {iteration} avec un coût de {cost}.")
            return True
        else:
            return False

    def predict(self, x):
        z = x.dot(self.w) + self.b
        return self.activation.predict(z)
