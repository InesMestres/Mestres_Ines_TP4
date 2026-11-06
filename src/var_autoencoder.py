
import torch
import torch.nn as nn

class Var_autoencoder(nn.Module):

    def __init__(self, dim_final):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 300), nn.ReLU())
        self.mean_layer = nn.Linear(300, dim_final)
        self.log_var_layer = nn.Linear(300, dim_final)
        self.decoder = nn.Sequential(nn.Linear(dim_final, 300), nn.ReLU(), nn.Linear(300, 784), nn.Sigmoid())

    def encode(self, X): 
        intermediate_values = self.encoder(X)
        mean = self.mean_layer(intermediate_values)
        log_variance = self.log_var_layer(intermediate_values)
        return mean, log_variance

    def latent_vector(self, mean, log_variance): 
        variance = torch.exp(log_variance)
        std_dev = torch.sqrt(variance)
        noise = torch.randn_like(std_dev)
        z = mean + std_dev * noise
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X): 
        mean, log_variance = self.encode(X)
        z = self.latent_vector(mean, log_variance)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_variance


def loss_VAE(original, reconstructed, mean, log_variance):
    squared_errors = (original - reconstructed) ** 2
    reconstruction_loss = 0.5 * squared_errors.sum(dim=1).mean()
    variance = torch.exp(log_variance)
    kl_terms = 1 + log_variance - mean ** 2 - variance
    kl_per_image = -0.5 * kl_terms.sum(dim=1)
    kl_loss = kl_per_image.mean()
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss
