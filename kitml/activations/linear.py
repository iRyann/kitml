from kitml.activations.activation import Activation

class Linear(Activation):
    THRESHOLD = 0.5

    def evaluate(self, x):
        return x

    def derivative(self, x):
        return 1
