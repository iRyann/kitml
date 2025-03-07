from abc import abstractmethod, ABC

class Activation(ABC) : 

    @abstractmethod
    def evaluate(self, x) :
        pass

