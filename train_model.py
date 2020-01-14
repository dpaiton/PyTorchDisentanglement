import os
import numpy as np
import argparse
import time as ti

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import params.param_loader as pl
import models.model_loader as ml
import utils.base_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="Path to the parameter file")

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = pl.load_param_file(param_file)


def load_dataset(params):
    if(params.dataset.lower() == "mnist"):
        # Load dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='../Datasets/', train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0,), std=(255,)), # rescale to be 0 to 1
            transforms.Lambda(lambda x: x.permute(1, 2, 0))])), # channels last
            batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='../Datasets/', train=False, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0,), std=(255,)), # rescale to be 0 to 1
            transforms.Lambda(lambda x: x.permute(1, 2, 0))])), # channels last
            batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    else:
        assert False, ("Supported datasets are ['mnist'], not"+dataset_name)
    params.epoch_size = len(train_loader.dataset)
    if(not hasattr(params, "num_val_images")):
        params.num_val_images = len(test_loader.dataset)
    if(not hasattr(params, "num_test_images")):
        params.num_test_images = len(test_loader.dataset)
    params.data_shape = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, params)


train_loader, val_loader, test_loader, params = load_dataset(params)


# Load model
model = ml.load_model(params.model_type)
model.setup(params)
model.to(params.device)


# Train model
for epoch in range(1, model.params.num_epochs+1):
   utils.train_epoch(epoch, model, train_loader)
   if(model.params.model_type.lower() in ["mlp", "ensemble"]):
       utils.test_epoch(epoch, model, test_loader)
   print("Completed epoch %g/%g"%(epoch, model.params.num_epochs))

# Checkpoint model
PATH = model.params.cp_save_dir
if not os.path.exists(PATH):
    os.makedirs(PATH)
SAVEFILE = PATH + "trained_model.pt"
torch.save(model.state_dict(), SAVEFILE)
