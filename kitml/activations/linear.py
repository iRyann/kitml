from activations.activation import Activation

class Linear(Activation):
    
    def evaluate(self, x):
        return x

    def derivative(self, x):
        return 1
