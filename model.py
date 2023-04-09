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
        self.x_embedder = nn.Linear(input_dim, 128)
        self.y_embedder = nn.Linear(input_dim, 128)
        self.r_embedder = nn.Linear(input_dim, 128)
        self.g_embedder = nn.Linear(input_dim, 128)
        self.b_embedder = nn.Linear(input_dim, 128)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_head, batch_first=True)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        x_coor = F.relu(self.x_embedder(x[:, 0, :]))
        y_coor = F.relu(self.y_embedder(x[:, 1, :]))
        r_value = F.relu(self.r_embedder(x[:, 2, :]))
        g_value = F.relu(self.g_embedder(x[:, 3, :]))
        b_value = F.relu(self.b_embedder(x[:, 4, :]))
        x_coor, y_coor, r_value, g_value, b_value = x_coor[:, None, :], y_coor[:, None, :], r_value[:, None, :], g_value[:, None, :], b_value[:, None, :]
        x = torch.cat((x_coor, y_coor, r_value, g_value, b_value), 1)
        x = self.transformer_layer(x)
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var

class Decoder(nn.Module):
    """
    Reconstruct the input by latent codes
    """
    def __init__(self, latent_dim, output_dim, num_head):
        super(Decoder, self).__init__()
        self.fc_emb = nn.Linear(latent_dim, 128)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_head, batch_first=True)
        self.fc_output = nn.Linear(128, output_dim)

    def forward(self, z):
        z = F.relu(self.fc_emb(z))
        z = self.transformer_layer(z)
        z = torch.sigmoid(self.fc_output(z))

        return z

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
        z = self.reparameterization(mean, torch.exp(log_var))
        return self.decoder(z), mean, log_var