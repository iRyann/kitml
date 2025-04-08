import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def gradientsForPerceptron(self):
        pass

    @abstractmethod
    def gradientsForSingleLayerPerceptron(self):
        pass


