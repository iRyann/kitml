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
        1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1
    ])

    y = y.reshape((y.shape[0], 1))

    return X, y

def dataset_2_10():
    X = np.array([
        [1, 2], [1, 4], [1, 5], [7, 5], [7, 6],
        [2, 1], [2, 3], [2, 4], [6, 2], [6, 4],
        [6, 5], [3, 1], [3, 2], [3, 4], [3, 5],
        [5, 3], [5, 4], [5, 6], [5, 7], [4, 2],
        [4, 3], [4, 5], [4, 6]
        ])

    y = np.array([
        1, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 0, 1, 1,
        0, 0, 1, 1, 0,
        1, 1, 1
        ])
    
    y = y.reshape((y.shape[0], 1))
    return X, y

def dataset_2_11():
    X = np.array([
        10, 14, 12, 18, 16, 14, 22, 28, 26, 16,
        23, 25, 20, 20, 24, 12, 15, 18, 14, 26,
        25, 17, 12, 20, 23, 22, 26, 22, 18, 21
    ]).reshape(-1, 1)

    y = np.array([
        4.4, 5.6, 4.6, 6.1, 6.0, 7.0, 6.8, 10.6, 11.0, 7.6,
        10.8, 10.0, 6.5, 8.2, 8.8, 5.5, 5.0, 8.0, 7.8, 9.0,
        9.4, 8.5, 6.4, 7.5, 9.0, 8.1, 8.2, 10.0, 9.1, 9.0
    ])

    y = y.reshape((y.shape[0], 1))
    return X, y