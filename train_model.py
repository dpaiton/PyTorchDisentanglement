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

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="Path to the parameter file")

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = pl.load_param_file(param_file)

def load_dataset(params):
    if params.dataset.lower() == "mnist":
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
    if not hasattr(params, "num_val_images"):
        params.num_val_images = len(test_loader.dataset)
    if not hasattr(params, "num_test_images"):
      params.num_test_images = len(test_loader.dataset)
    params.data_shape = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, params)

train_loader, val_loader, test_loader, params = load_dataset(params)

# Load model
model = ml.load_model(params.model_type)
model.setup(params)
model.to(params.device)

# Setup optimizer
if(params.optimizer.name == "sgd"):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params.weight_lr,
        weight_decay=params.weight_decay)
elif params.optimizer.name == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.weight_lr,
        weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=params.optimizer.milestones,
    gamma=params.optimizer.lr_decay_rate)


# Define train & test functions
def train(epoch, params):
    model.train()
    epoch_size = len(train_loader.dataset)
    num_batches = epoch_size / params.batch_size
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(params.device), target.to(params.device)
        optimizer.zero_grad() # clear gradietns of all optimized variables
        loss = model.get_total_loss((data, target))
        loss.backward() # backward pass
        optimizer.step()
        if(hasattr(params, "renormalize_weights") and params.renormalize_weights):
            with torch.no_grad():
                model.w.div_(torch.norm(model.w, dim=0, keepdim=True))
        if(batch_idx % int(num_batches/params.train_logs_per_epoch) == 0.):
            batch_step = epoch * model.params.batches_per_epoch + batch_idx
            model.print_update(input_data=data, input_labels=target, batch_step=batch_step)
    scheduler.step(epoch)


def test(epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(params.device), target.to(params.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        stat_dict = {
            "epoch":epoch,
            "test_loss":test_loss,
            "test_correct":correct,
            "test_total":len(test_loader.dataset),
            "test_accuracy":test_accuracy}
        js_str = model.js_dumpstring(stat_dict)
        model.log_info("<stats>"+js_str+"</stats>")

# Train model
for epoch in range(1, params.num_epochs+1):
    train(epoch, params)
    if(params.model_type == "mlp"):
        test(epoch)
    print("Completed epoch %g/%g"%(epoch, params.num_epochs))


# Checkpoint model
PATH = model.params.cp_save_dir#"../Projects/"+params.model_type+"/savefiles/"
if not os.path.exists(PATH):
    os.makedirs(PATH)
SAVEFILE = PATH + "trained_model.pt"
torch.save(model.state_dict(), SAVEFILE)
