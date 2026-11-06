import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, dim_final):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 300), nn.ReLU(), nn.Linear(300, dim_final))
        self.decoder = nn.Sequential(nn.Linear(dim_final, 300), nn.ReLU(), nn.Linear(300, 784), nn.Sigmoid())
    
    def forward(self, X):
        Z = self.encoder(X)
        X_reconstructed = self.decoder(Z)
        return X_reconstructed
        