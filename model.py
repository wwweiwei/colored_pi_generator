import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    """
    Encodes the input by passing through the encoder
    """
    def __init__(self, input_dim, latent_dim, num_head):
        super(VariationalEncoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, 256)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_head, batch_first=True)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=0)
        x = F.relu(self.fc_input(x))
        x = self.transformer_layer(x.unsqueeze(0).unsqueeze(0))
        mean = self.fc_mean(x.squeeze())
        log_var = self.fc_var(x.squeeze())
        return mean, log_var

class Decoder(nn.Module):
    """
    Reconstruct the input by latent codes
    """
    def __init__(self, latent_dim, output_dim, num_head):
        super(Decoder, self).__init__()
        self.fc_emb = nn.Linear(latent_dim, 256)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_head, batch_first=True)
        self.fc_output = nn.Linear(256, output_dim)

    def forward(self, z):
        z = F.relu(self.fc_emb(z))
        z = self.transformer_layer(z.unsqueeze(0).unsqueeze(0))
        z = torch.sigmoid(self.fc_output(z))

        return z.reshape((5000, 5))

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_head):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim, num_head)
        self.decoder = Decoder(latent_dim, output_dim=input_dim, num_head=num_head)

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