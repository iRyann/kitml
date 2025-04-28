import matplotlib.pyplot as plt
from kitml.utilities.dataset import *
from kitml.models.perceptron import Perceptron

from kitml.metrics.logLoss import LogLoss
from kitml.metrics.meanQuadraticError import MeanQuadraticError

from kitml.activations.sigmoid import Sigmoid
from kitml.activations.linear import Linear

import numpy as np

def test_perceptron_and():
    x_train, y_train = and_set()
    perceptron = Perceptron(input_size=x_train.shape[1], metric=LogLoss(), eta=0.5, nb_epoch=1000, a=Sigmoid())
    cost, acc = perceptron.train(x_train, y_train, error_threshold=0.01)

    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)

    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('outputs/perceptron_and_training.png')
    plt.show()

    print(f"Sur l'entrée \n {x_train},\n le Perceptrone prédit bien les classes :\n {perceptron.predict(x_train)}")
    print("Toutefois, on peut visualiser la séparation linéaire opérée par le perceptron.")

    f = lambda x: - (perceptron.w[0] * x + perceptron.b) / perceptron.w[1]
    x_values = np.linspace(-0.5, 1.5, 200)

    plt.scatter(x_train[:, 0], x_train[:, 1], label='Données')
    plt.plot(x_values, f(x_values), label='Ligne de décision', color='red')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Ligne de décision du perceptron")
    plt.savefig('outputs/perceptron_and_decision_boundary.png')
    plt.show()

def test_perceptron_2_9():
    x_train, y_train = dataset_2_9()
    perceptron = Perceptron(input_size=x_train.shape[1], metric=LogLoss(), eta=0.5, nb_epoch=1000, a=Sigmoid())
    cost, acc = perceptron.train(x_train, y_train, error_threshold=0.01)
    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/perceptron_2_9_training.png')
    plt.show()
    # print(f"Sur l'entrée \n {x_train},\n le Perceptrone prédit bien les classes :\n {perceptron.predict(x_train)}")
    print("Toutefois, on peut visualiser la séparation linéaire opérée par le perceptron.")
    f = lambda x: - (perceptron.w[0] * x + perceptron.b) / perceptron.w[1]
    x_values = np.linspace(-0.5, 8 , 200)
    plt.scatter(x_train[:, 0], x_train[:, 1], label='Données')
    plt.plot(x_values, f(x_values), label='Ligne de décision', color='red')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Ligne de décision du perceptron")
    plt.savefig('outputs/perceptron_2_9_decision_boundary.png')
    plt.show()


def test_perceptron_2_10():
    x_train, y_train = dataset_2_10()
    perceptron = Perceptron(input_size=x_train.shape[1], metric=LogLoss(), eta=0.015, nb_epoch=1000, a=Sigmoid())
    cost, acc = perceptron.train(x_train, y_train, error_threshold=0.01)
    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/perceptron_2_10_training.png')
    plt.show()
    # print(f"Sur l'entrée \n {x_train},\n le Perceptrone prédit bien les classes :\n {perceptron.predict(x_train)}")
    print("Toutefois, on peut visualiser la séparation linéaire opérée par le perceptron.")
    f = lambda x: - (perceptron.w[0] * x + perceptron.b) / perceptron.w[1]
    x_values = np.linspace(-0.5, 8 , 200)
    cmap = plt.get_cmap('coolwarm')
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.flatten(), cmap=cmap, label='Données')
    plt.plot(x_values, f(x_values), label='Ligne de décision', color='red')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Ligne de décision du perceptron")
    plt.savefig('outputs/perceptron_2_10_decision_boundary.png')
    plt.show()


def test_perceptron_2_11():
    x_train, y_train = dataset_2_11()
    perceptron = Perceptron(input_size=x_train.shape[1], metric=MeanQuadraticError(), eta=0.0015, nb_epoch=1000, a=Linear())
    cost, acc = perceptron.train(x_train, y_train, error_threshold=0.01)
    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/perceptron_2_11_training.png')
    plt.show()
    
    # Affichage de la régression linéaire, de la courbe prédite
    x_values = np.linspace(7, 30, 200)
    plt.scatter(x_train, y_train, label='Données')
    y_pred = np.array([])
    for x in x_values:
        y_pred = np.append(y_pred, perceptron.predict(np.array([[x]])))
    y_pred = np.array(y_pred).flatten()
    plt.plot(x_values, y_pred, label='Ligne de régression', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.title("Ligne de régression du perceptron")
    plt.savefig('outputs/perceptron_2_11_decision_boundary.png')
    plt.show()




# test_perceptron_and()
# test_perceptron_2_9()
# test_perceptron_2_10()
# test_perceptron_2_11()



