from sklearn.datasets import make_blobs

def simple_linear_set() :
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    print(y.shape)
    y = y.reshape((y.shape[0], 1))
    print(y.shape)
    return X,y

simple_linear_set()