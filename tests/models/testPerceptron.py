import matplotlib.pyplot as plt
from kitml.utilities.dataset import and_set
from kitml.models.perceptron import Perceptron
from kitml.metrics.logLoss import LogLoss
from kitml.activations.sigmoid import Sigmoid
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
    plt.show()

test_perceptron_and()
