from abc import abstractmethod, ABC
import numpy as np
class Activation(ABC) : 
    
    THRESHOLD = 0.

    @abstractmethod
    def evaluate(self, x) :
        pass
    
    @abstractmethod
    def derivative(self, x):
        pass
    
    def predict(self, z, evaluate = True):
        res = z
        
        if evaluate:
            res = self.evaluate(res)
    
        return np.where(res >= self.THRESHOLD, 1, 0)

