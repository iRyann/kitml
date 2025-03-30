from abc import abstractmethod, ABC
import numpy as np
class Activation(ABC) : 

    @abstractmethod
    def evaluate(self, x) :
        pass
    
    @abstractmethod
    def derivative(self, x):
        pass
    
    def predict(self, z, threshold, evaluate = True):
        res = z
        
        if evaluate:
            res = self.evaluate(res)
    
        return np.where(res >= threshold, 1, 0)

