import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    """
    Encodes the input by passing through the encoder
    """
    def __init__(self, input_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, 512)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=0)
        x = F.relu(self.fc_input(x))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var

class Decoder(nn.Module):
    """
    Reconstruct the input by latent codes.
    """
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((5000, 5))

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim=input_dim)

    def reparameterization(self, mean, var):
        """
        Sample the epsilon from standard normal distribution with the size same as var
        and get the latent representation (i.e. z) by reparameterization trick
        """
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5*log_var))
        return self.decoder(z), mean, log_var