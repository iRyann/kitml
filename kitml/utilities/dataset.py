from sklearn.datasets import make_blobs
import numpy as np

def simple_linear_set() :
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    return X,y

def and_set() :
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1]).reshape(4, 1)
    return X,y

def xor_set():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0]).reshape(4, 1)
    return X,y

def dataset_2_9():
    X = np.array([
        [1, 6],
        [7, 9],
        [1, 9],
        [7, 10],
        [2, 5],
        [2, 7],
        [2, 8],
        [6, 8],
        [6, 9],
        [3, 5],
        [3, 6],
        [3, 8],
        [3, 9],
        [5, 7],
        [5, 8],
        [5, 10],
        [5, 11],
        [4, 6],
        [4, 7],
        [4, 9],
        [4, 10]
    ])

    y = np.array([
        1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1
    ])

    y = y.reshape((y.shape[0], 1))

    return X, y