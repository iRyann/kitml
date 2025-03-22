import matplotlib.pyplot as plt
from kitml.utilities.dataset import and_set
from kitml.models.neuron import Neuron
from kitml.metrics.logLoss import LogLoss
from kitml.activations.sigmoid import Sigmoid
import numpy as np

def testNeuronAnd():
    x_train, y_train = and_set()
    neuron = Neuron(x_train, y_train, x_train, y_train, LogLoss(), 0.5, 1000, Sigmoid())
    cost, acc = neuron.train()

    # Tracer le coût
    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype = int)
    
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()

    # Tracer la précision
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(neuron.predict(x_train))

testNeuronAnd()
