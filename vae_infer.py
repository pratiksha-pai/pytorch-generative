import torch
import pytorch_generative as pgw
from pytorch_generative.models.autoregressive.nade import NADE, reproduce
from pytorch_generative.models.vae.vae import VAE, reproduce
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer

log_dir = "/home/hice1/ppai33/scratch/vae_run1/"
checkpoint_path = log_dir + "trainer_state_50.ckpt"
checkpoint = torch.load(checkpoint_path)

model = VAE(
    in_channels=1,
    out_channels=1,
    latent_channels=16,
    strides=[2, 2, 2, 2],
    hidden_channels=64,
    residual_channels=32,
)

model.load_state_dict(checkpoint["model"])
model.eval()

# Sample from the model
samples = model.sample(n_samples=5)
samples = np.squeeze(samples)
if isinstance(samples, torch.Tensor):
    samples = samples.detach().cpu().numpy()

samples = (samples * 255).astype(np.uint8)
combined_image = np.hstack(samples)
plt.imsave("vae.png", combined_image, cmap='gray')

