import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from kitml.activations.softMax import SoftMax
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class SingleLayerPerceptron:
    rgn = np.random.default_rng(25)

    def __init__(self, input_size, output_size, a: Activation, metric: Metric, eta, nb_epoch):
        self.w = SingleLayerPerceptron.rgn.random(size=(output_size, input_size))
        self.b = SingleLayerPerceptron.rgn.random(size=(output_size, 1))
        self.m = metric 
        self.eta = eta
        self.nb_iter = nb_epoch
        self.activation = a
        self.output_size = output_size

    def model(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        z = self.w @ x.T + self.b
        return self.activation.evaluate(z)

    def update(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def train(self, x_train, y_train, error_threshold=0.01):
        cost_values = []
        accuracy_values = []

        # TODO : Gestion explicite du type d'opération (classification ou regression)
        if isinstance(self.activation, SoftMax) and (len(y_train.shape) == 1 or y_train.shape[1] == 1):
            y_train_one_hot = np.zeros((y_train.shape[0], self.output_size))
            for i, y in enumerate(y_train):
                y_train_one_hot[i, int(y)] = 1
            y_train = y_train_one_hot
        elif len(y_train.shape) > 1 and y_train.shape[1] != self.output_size:
            raise ValueError(f"Le nombre de labels de y_train ({y_train.shape[1]}) ne correspond pas à output_size ({self.output_size}).")
    

        for i in tqdm(range(self.nb_iter)):
            a = self.model(x_train)  # a : (output_size, n_samples)
            dw, db = self.m.gradientsForSingleLayerPerceptron(y_train.T, a, x_train)
            self.update(dw, db)

            if i % 10 == 0:
                end = self._evaluate_and_check_convergence(y_train.T, a, i, cost_values, accuracy_values, error_threshold)
                if end:
                    break

        return cost_values, accuracy_values

    def _evaluate_and_check_convergence(self, y_train, a, iteration, cost_values, accuracy_values, error_threshold):
        cost = self.m.evaluate(y_train, a)
        cost_values.append(cost)

        y_pred = np.argmax(a, axis=0)
        y_true = np.argmax(y_train, axis=0)
        accuracy = np.mean(y_pred == y_true)
        accuracy_values.append(accuracy)

        if cost < error_threshold:
            print(f"Convergence atteinte à l'itération {iteration} avec un coût de {cost}.")
            return True
        return False

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
    
        z = self.w @ x.T + self.b
        a = self.activation.evaluate(z)
        
        if isinstance(self.activation, SoftMax):
            return np.argmax(a, axis=0)
        else:
            return a.T