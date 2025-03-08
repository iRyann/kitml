import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    
    @abstractmethod
    @staticmethod
    def evaluate():
        pass

    @abstractmethod
    @staticmethod
    def gradientsForNeuron():
        pass


