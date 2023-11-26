import torch
import pytorch_generative as pgw
from pytorch_generative.models.vae.vae import VAE, reproduce
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer

# Set parameters
log_dir = "/home/hice1/ppai33/scratch/vae_run1"

def reproduce(
    n_epochs=457,
    batch_size=128,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):

    import torch
    from torch import optim
    from torch.nn import functional as F

    from pytorch_generative import datasets, models, trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True, resize_to_32=True
        )

    model = models.VAE(
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        strides=[2, 2, 2, 2],
        hidden_channels=64,
        residual_channels=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn(x, _, preds):
        preds, kl_div = preds
        recon_loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        recon_loss = recon_loss.sum(dim=(1, 2, 3))
        elbo = recon_loss + kl_div

        return {
            "recon_loss": recon_loss.mean(),
            "kl_div": kl_div.mean(),
            "loss": elbo.mean(),
        }

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    model_trainer.interleaved_train_and_eval(n_epochs)

samples = reproduce(n_epochs=50, batch_size=256, log_dir=log_dir)


