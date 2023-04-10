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
        
        self.fc_mean_x = nn.Linear(128, latent_dim)
        self.fc_mean_y = nn.Linear(128, latent_dim)
        self.fc_mean_r = nn.Linear(128, latent_dim)
        self.fc_mean_g = nn.Linear(128, latent_dim)
        self.fc_mean_b = nn.Linear(128, latent_dim)

        self.fc_var_x = nn.Linear(128, latent_dim)
        self.fc_var_y = nn.Linear(128, latent_dim)
        self.fc_var_r = nn.Linear(128, latent_dim)
        self.fc_var_g = nn.Linear(128, latent_dim)
        self.fc_var_b = nn.Linear(128, latent_dim)

    def forward(self, x):
        x_coor = F.relu(self.x_embedder(x[:, 0:1, :]))
        y_coor = F.relu(self.y_embedder(x[:, 1:2, :]))
        r_value = F.relu(self.r_embedder(x[:, 2:3, :]))
        g_value = F.relu(self.g_embedder(x[:, 3:4, :]))
        b_value = F.relu(self.b_embedder(x[:, 4:5, :]))

        x = torch.cat((x_coor, y_coor, r_value, g_value, b_value), 1)
        x = self.transformer_layer(x)

        mean_x, mean_y, mean_r, mean_g, mean_b = self.fc_mean_x(x[:, 0:1, :]), self.fc_mean_y(x[:, 1:2, :]), self.fc_mean_r(x[:, 2:3, :]), self.fc_mean_g(x[:, 3:4, :]), self.fc_mean_b(x[:, 4:5, :])
        log_var_x, log_var_y,log_var_r, log_var_g, log_var_b = self.fc_var_x(x[:, 0:1, :]), self.fc_var_y(x[:, 1:2, :]), self.fc_var_r(x[:, 2:3, :]), self.fc_var_g(x[:, 3:4, :]), self.fc_var_b(x[:, 4:5, :])

        mean = torch.cat((mean_x, mean_y, mean_r, mean_g, mean_b), 1)
        log_var = torch.cat((log_var_x, log_var_y,log_var_r, log_var_g, log_var_b), 1)

        return mean, log_var

class Decoder(nn.Module):
    """
    Reconstruct the input by latent codes
    """
    def __init__(self, latent_dim, output_dim, num_head):
        super(Decoder, self).__init__()
        self.x_embedder = nn.Linear(latent_dim, 128)
        self.y_embedder = nn.Linear(latent_dim, 128)
        self.r_embedder = nn.Linear(latent_dim, 128)
        self.g_embedder = nn.Linear(latent_dim, 128)
        self.b_embedder = nn.Linear(latent_dim, 128)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_head, batch_first=True)
        
        self.fc_output_x = nn.Linear(128, 128)
        self.fc_output_y = nn.Linear(128, 128)
        self.fc_output_r = nn.Linear(128, 128)
        self.fc_output_g = nn.Linear(128, 128)
        self.fc_output_b = nn.Linear(128, 128)
        
        self.fc_output = nn.Linear(128, output_dim)

    def forward(self, z):
        x_coor = F.relu(self.x_embedder(z[:, 0:1, :]))
        y_coor = F.relu(self.y_embedder(z[:, 1:2, :]))
        r_value = F.relu(self.r_embedder(z[:, 2:3, :]))
        g_value = F.relu(self.g_embedder(z[:, 3:4, :]))
        b_value = F.relu(self.b_embedder(z[:, 4:5, :]))

        z = torch.cat((x_coor, y_coor, r_value, g_value, b_value), 1)
        z = self.transformer_layer(z)

        z_x, z_y, z_r, z_g, z_b = F.relu(self.fc_output_x(z[:, 0:1, :])), F.relu(self.fc_output_y(z[:, 1:2, :])), F.relu(self.fc_output_r(z[:, 2:3, :])), F.relu(self.fc_output_g(z[:, 3:4, :])), F.relu(self.fc_output_b(z[:, 4:5, :])) 

        output = F.sigmoid(self.fc_output(torch.cat((z_x, z_y, z_r, z_g, z_b), 1)))
        return output

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