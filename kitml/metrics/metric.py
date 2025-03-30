import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def gradientsForPerceptron(self):
        pass


