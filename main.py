from tqdm import tqdm
import torch
import torch.utils
import torch.distributions
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import PiDataset
from model import VariationalAutoencoder
from utils import *

class PiGenerator:
    def __init__(self, model, latent_dim, epochs, result_path, num_pixel):
        super(PiGenerator, self).__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.result_path = result_path
        self.num_pixel = num_pixel
        self.z = torch.randn(latent_dim).float().to(device)

    def loss_function(self, x, x_hat, mean, log_var):
        """
        Calculate the reconstruction loss and KLD loss
        """
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = - 0.5 * torch.sum(1+log_var-mean.pow(2)-log_var.exp())
        return recons_loss, kld_loss

    def train(self, data, gen_every_epochs):
        """
        Train the VAE by update the weights iteratively
        """
        opt = torch.optim.Adam(self.model.parameters())
        total_loss, total_kl, total_rc, beta = [], [], [], 0
        pbar = tqdm(range(self.epochs), desc='Epoch: ')
        for epoch in pbar:
            for _, x in enumerate(data):
                x = x.float().to(device)
                opt.zero_grad()
                x_hat, mean, log_var = self.model(x.float())

                recons_loss, kld_loss = self.loss_function(x.squeeze(), x_hat.squeeze(), mean, log_var)
                loss = (1-beta) * recons_loss + beta * kld_loss
                loss.backward()
                opt.step()

                cur_total_loss = loss.item()
                cur_kl_loss = kld_loss.item()
                cur_rc_loss = recons_loss.item()
                total_loss.append(cur_total_loss)
                total_kl.append(cur_kl_loss)
                total_rc.append(cur_rc_loss)

                pbar.set_description('Total Loss: {}| KLD Loss: {}| RC Loss: {}'.format(round(cur_total_loss, 2), round(cur_kl_loss, 2), round(cur_rc_loss, 2)), refresh=True)
                if gen_every_epochs:
                    self.generator(epoch)
        
            beta += 1/self.epochs
        return total_loss, total_kl, total_rc

    def generator(self, cur_epoch):
        """
        Generate the image by random sampling the latent representation and putting into the decoder
        """
        self.model.eval()  
        with torch.no_grad():
            outputs = self.model.decoder(self.z).squeeze().cpu().numpy()

        x_coor, y_coor = xy_rescaling(xy_coor=outputs[:, 0:2])
        rgb_values = rgb_rescaling(rgb_values=outputs[:, 2:5])

        generate_img(x_coor, y_coor, rgb_values, self.result_path, cur_epoch)


if __name__ == '__main__':
    latent_dim, epochs, batch_size, device, result_path, gen_every_epochs, num_workers, retrain, num_head = get_config()
    
    # Load the data
    data = PiDataset()
    pi_dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Call the class to construct an object
    model = VariationalAutoencoder(input_dim=data.get_total_num_of_data(), latent_dim=latent_dim, num_head=num_head).to(device)
    print(f'model: {model}')
    pi_generator = PiGenerator(model=model, latent_dim=latent_dim, epochs=epochs, result_path=result_path, num_pixel=data.get_total_num_of_pixel())
    
    if retrain:
        # Train VAE
        total_loss, total_kl, total_rc = pi_generator.train(data=pi_dataloader, gen_every_epochs=gen_every_epochs)
        draw_loss_curve(total_num_epoch=epochs, total_loss=total_loss, total_kl=total_kl, total_rc=total_rc, result_path=result_path)
        
        # Save the model weight
        torch.save(model.state_dict(), './vae.pth')
    else:
        # Load the model weight
        model.load_state_dict(torch.load('./vae.pth'))
    
    # Generate the image
    pi_generator.generator(cur_epoch=epochs)