import torch
import pytorch_generative as pgw
from pytorch_generative.models.autoregressive.nade import NADE, reproduce
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer
from torch import optim
from torch.nn import functional as F
from pytorch_generative import datasets, models, trainer


n_epochs = 30
hidden_dims = 300
log_dir = "/home/hice1/ppai33/scratch/made_run1" 

def reproduce(
    n_epochs=85,
    batch_size=64,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """


    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True
        )

    model = models.MADE(input_dim=784, hidden_dims=[hidden_dims], n_masks=1)
    optimizer = optim.Adam(model.parameters())

    def loss_fn(x, _, preds):
        batch_size = x.shape[0]
        x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        return loss.sum(dim=1).mean()

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


reproduce(n_epochs=n_epochs, log_dir=log_dir)