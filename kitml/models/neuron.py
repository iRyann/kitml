import numpy as np
from kitml.metrics.metric import Metric
from kitml.activations.activation import Activation
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# TODO Retirer les trains et test set au profit de paramètre de méthode
# TODO Ajouter le seuil en paramètre (relatif à la fonction d'activation !!)
class Neuron :
    rgn = np.random.default_rng(25)

    def __init__(self, x_train, y_train, x_test, y_test, metric : Metric, eta, nb_epoch, a : Activation):
        self.w = np.reshape(np.array(Neuron.rgn.uniform(size = x_train.shape[1])), shape = (x_train.shape[1],1))
        self.b = Neuron.rgn.uniform()
        self.m = metric
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.eta = eta
        self.nb_iter = nb_epoch
        self.activation = a

    def model(self) :
        z = self.x_train.dot(self.w) + self.b
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
                b = np.reshape(a,(4,1)).T
                cost_values.append(self.m.evaluate(self.y_train, b))
                y_pred = self.activation.predict(b, 0.5, False)
                accuracy_values.append(accuracy_score(self.y_train, y_pred))

            dw, db = self.m.gradientsForNeuron(self.y_train, a, self.x_train)
            self.update(dw, db)
        
        return cost_values, accuracy_values
        
    def predict(self, x):
        z = x.dot(self.w) + self.b
        return self.activation.predict(z, 0.5, True)