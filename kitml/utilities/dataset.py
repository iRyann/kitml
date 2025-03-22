from sklearn.datasets import make_blobs
import numpy as np

def simple_linear_set() :
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    return X,y

def and_set() :
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1]).reshape(4, 1)
    return X,y.T

def xor_set():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,1]).reshape(4, 1)
    return X,y