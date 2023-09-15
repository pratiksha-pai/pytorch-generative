import torch
import pytorch_generative as pgw
from pytorch_generative.models.autoregressive.nade import NADE, reproduce
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer


log_dir = "/home/hice1/ppai33/scratch/made_run1/"  
checkpoint_path = log_dir + "trainer_state_30.ckpt"
checkpoint = torch.load(checkpoint_path)
hidden_dims = 300
input_dim = 784  
model = models.MADE(input_dim=input_dim, hidden_dims=[hidden_dims])
model.load_state_dict(checkpoint["model"])
model.eval()

# Sample from the model
samples = model.sample(n_samples=5)
samples = np.squeeze(samples)
if isinstance(samples, torch.Tensor):
    samples = samples.detach().cpu().numpy()

samples = (samples * 255).astype(np.uint8)
combined_image = np.hstack(samples)
plt.imsave("made_combined_image.png", combined_image, cmap='gray')

