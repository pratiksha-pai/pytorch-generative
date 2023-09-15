import torch
import pytorch_generative as pgw
from pytorch_generative.models.autoregressive.nade import NADE, reproduce

# Set parameters
epochs = 30
hidden_dim = 300
batch_size = 128  # or any other size you want
logdir = "/home/hice1/ppai33/scratch/run"  # or any other directory for TensorBoard logs
n_gpus = 1  # or any other number of GPUs you want to use

# Initialize NADE model
input_dim = 784  # This should be set based on your specific data
nade_model = NADE(input_dim=input_dim, hidden_dim=hidden_dim)

# Train the model
reproduce(epochs, batch_size, logdir, n_gpus)

# Optional: Generate and visualize samples (assuming sample() exists in your model)
samples = nade_model.sample(num_samples=5)
print(samples)
