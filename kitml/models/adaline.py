import numpy as np
from kitml.activations.activation import Activation
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Adaline:
    """
    Adaline (Adaptive Linear Neuron) classifier.

    Attributes:
        rgn (np.random.Generator): Random number generator.
        w (np.ndarray): Weights after fitting.
        x_train (np.ndarray): Training data with bias term.
        x_test (np.ndarray): Test data with bias term.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        eta (float): Learning rate.
        nb_iter (int): Number of epochs.
        activation (Activation): Activation function.

    Methods:
        model(x):
            Compute the linear activation for input x.

        train():
            Train the Adaline model using the training data.
            Returns:
                cost_values (list): List of cost values over epochs.
                accuracy_values (list): List of accuracy values over epochs.

        predict(x):
            Predict class labels for samples in x.
            Returns:
                np.ndarray: Predicted class labels.
    """
    rgn = np.random.default_rng(25)

    def __init__(self, x_train, y_train, x_test, y_test, eta, nb_epoch, a: Activation):
        self.w = Adaline.rgn.uniform(size=x_train.shape[1] + 1)
        self.x_train = np.c_[x_train, np.ones(x_train.shape[0])]  
        self.x_test = np.c_[x_test, np.ones(x_test.shape[0])]
        self.y_train = y_train
        self.y_test = y_test
        self.eta = eta
        self.nb_iter = nb_epoch
        self.activation = a

    def model(self, x):
        z = np.dot(x, self.w)
        return z  # Activation linéaire

    def train(self):
        cost_values = []
        accuracy_values = []

        for i in tqdm(range(self.nb_iter)):
            activation_list = []

            for j in range(len(self.x_train)):
                x = self.x_train[j]
                y = self.y_train[j]
                a = self.model(x)
                error = a - y
                dw = error * x
                self.w -= self.eta * dw 

                if i % 10 == 0 :
                    activation_list.append(a)

            if i % 10 == 0:
                activation_list = np.array(activation_list)
                cost_values.append(np.mean((activation_list - self.y_train) ** 2))
                accuracy_values.append(accuracy_score(self.y_train, np.where(activation_list >= 0, 1, 0)))

        return cost_values, accuracy_values

    def predict(self, x):
        z = np.dot(x, self.w)
        return np.where(z >= 0, 1, 0)
