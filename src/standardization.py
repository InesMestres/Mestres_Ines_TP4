import numpy as np

def standardization_fit(X):
    means = np.mean(X, axis=0)
    devs = np.std(X, axis=0)
    devs[devs == 0] = 1
    return means, devs

def standardize(X, means, devs): 
    X_standard = (X - means) / devs
    return X_standard
