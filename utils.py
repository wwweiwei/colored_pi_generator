import random
import yaml
import numpy as np
import torch
from matplotlib.pylab import plt
from PIL import Image, ImageDraw

def set_seeds(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_config(config_filename='config.yaml'):
    """
    Read all the hyperparameters by config file
    """
    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)
    latent_dim = config['latent_dim']
    epochs = config['epochs']
    batch_size = config['batch_size']
    device = config['device']
    result_path = config['result_path']
    gen_every_epochs = config['gen_every_epochs']
    num_workers = config['num_workers']
    retrain = config['retrain']
    num_head = config['num_head']
    seed = config['seed']
    set_seeds(seed)

    return latent_dim, epochs, batch_size, device, result_path, gen_every_epochs, num_workers, retrain, num_head

def draw_loss_curve(total_num_epoch, total_loss, total_kl, total_rc, result_path):
    """
    Draw the training loss curve with total loss, KLD loss and reconstruction loss
    and save the plot as loss.png
    """
    epochs = range(1, total_num_epoch+1)
    plt.plot(epochs, total_loss, label='Total Loss')
    plt.plot(epochs, total_kl, label='KL Divergence Loss')
    plt.plot(epochs, total_rc, label='Reconstruction Loss')

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, total_num_epoch+1, 5))
    plt.legend(loc='best')
    plt.savefig(result_path+'loss.png')

def xy_rescaling(xy_coor):
    """
    Rescale the x-coordinates and y-coordinates into the range of (0, 300)
    """
    x_coor, y_coor = (xy_coor[:, 0, :]*300).astype(int), (xy_coor[:, 1, :]*300).astype(int)
    return np.reshape(x_coor, -1), np.reshape(y_coor, -1)

def rgb_rescaling(rgb_values):
    """
    Rescale the rgb values into the range of (0, 255)
    """
    r_value, g_value, b_value = (rgb_values[:, 0, :]*255.0).astype(int), (rgb_values[:, 1, :]*255.0).astype(int), (rgb_values[:, 2, :]*255.0).astype(int)
    return np.reshape(r_value, -1), np.reshape(g_value, -1), np.reshape(b_value, -1)

def generate_img(x_coor, y_coor, r_value, g_value, b_value, result_path, epochs):
    """
    Visualize the image by the x-coordinates, y-coordinates and rgb values generated by model
    and save the image as ep_#epochs.png
    """
    img = Image.new('RGB', (300, 300))
    draw = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(zip(x_coor, y_coor)):
        draw.point((y, x), fill=tuple([r_value[i], g_value[i], b_value[i]]))
    img.save(result_path+'ep_'+str(epochs)+'.png')