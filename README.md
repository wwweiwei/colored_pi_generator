# Image Generator Implemented by VAE
## Problem Formulation
Build a generative model `colored_pi_generator` whose output is a 5-dimensional point which comes from the **same distribution** as the points in the dataset.
* **Input**
    * pi_xs.npy - A numpy array of length 5000 - the x-coordinates of each of the points in the dataset
    * pi_ys.npy - A numpy array of length 5000 - the y-coordinates of each of the points in the dataset
    * An image
* **Output**
    * An image which comes from the same distribution as the points in the dataset

## Method - VAE
* Model Architecture
    * Encoder
    * Generator
* Loss Function
    * Reconstruction loss: how well VAE can reconstruct the input from the latent space
    * KLD Loss (Kullback–Leibler Divergence): how similar the latent distribution and standard normal distribution are
    * Beta: A weight to alleviate the KL-vanishing issue is to apply annealing schedules for the KLD loss
    * [Reference](https://medium.com/mlearning-ai/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023)

## Results
* Training Curves
    * Vanilla VAE    
    * VAE with **Attention Layer**

* Generate Results
    * Vanilla VAE
    * VAE with **Attention Layer**

## Usage
* Build environment
    ```
    conda env create -f environment.yml
    ```
* Put data in `./gen_ml_quiz_content/`
* Run script
    ```python
        python main.py
    ```

## Dataset
* Training data

## Hyperparameter Setting
* All the experiments are done by the hyperparameters below. Feel free to set your own hyperparameter in `config.yaml`
    | epochs | batch\_size | hidden\_dim | device | num\_workers | result\_path | gen\_every\_epochs | seed | retrain |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | 100 | 1 | 256 | 'cpu' | 4 | './result/' | True | 10 | True |

## Reference
* [Cyclical KLAnnealing Schedule](https://github.com/haofuml/cyclical_annealing)
* [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
* [Pytorch-VAE-tutorial](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial)