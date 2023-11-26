import torch
import pytorch_generative as pgw
from pytorch_generative.models.flow.nice import NICE
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer
from torch import optim
from torch.nn import functional as F

# Set parameters
input_dim = 784
hidden_dim = 300
log_dir = "/home/hice1/ppai33/scratch/nice_run1"


def reproduce(
    n_epochs=150,
    batch_size=1024,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    
    from pytorch_generative import datasets, models, trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dequantize=True
        )

    model = models.NICE(
        n_features=784, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    )
    # NOTE: We found most hyperparameters from the paper give bad results so we only use
    # the learning rate.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(x, _, preds):
        preds, log_det_J = preds
        log_prob = -(F.softplus(preds) + F.softplus(-preds)).sum(dim=(1, 2, 3))
        loss = log_prob + log_det_J
        return {
            "loss": -loss.mean(),
            "prior_log_likelihood": log_prob.mean(),
            "log_det_J": log_det_J.mean(),
        }

    trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        lr_scheduler=None,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    trainer.interleaved_train_and_eval(n_epochs)

samples = reproduce(n_epochs=50, batch_size=256, log_dir=log_dir)

