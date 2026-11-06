import numpy as np

def pca_fit(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    n = len(X_centered)
    cov = (X_centered.T @ X_centered)/n
    autovalores, autovectores = np.linalg.eigh(cov)
    order = np.argsort(autovalores)[::-1]
    autovalores = autovalores[order]
    autovectores = autovectores[:, order]
    return mean, autovalores, autovectores

def pca(X, mean, autovectores, n): 
    X_centered = X - mean
    components = autovectores[:,:n]
    Z = X_centered @ components
    return Z

def reconstruction_pca(Z, mean, autovectores, n): 
    components = autovectores[:, :n]
    X_reconstructed = Z @ components.T + mean
    return X_reconstructed
