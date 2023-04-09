import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class PiDataset(Dataset):
    def __init__(self):
        self.xs = np.load('./gen_ml_quiz_content/pi_xs.npy') # xs.shape is (5000,)
        self.ys = np.load('./gen_ml_quiz_content/pi_ys.npy') # ys.shape is (5000,)
        self.cor = np.concatenate([self.xs.reshape(len(self.xs), 1), self.ys.reshape(len(self.ys), 1)], axis=1)
        
        image_array = np.array(Image.open('./gen_ml_quiz_content/sparse_pi_colored.jpg'))
        self.rgb_values = image_array[self.xs, self.ys] # rgb_values.shape is (5000,3)
        self.data = np.concatenate([self.cor/300.0, self.rgb_values/255.0], axis=1)
        self.data = np.expand_dims(self.data, axis=-1)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.shape[0]
    
    def get_seq_len(self):
        return self.data.shape[1]