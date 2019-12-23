import os
import numpy as np
import argparse
import time as ti

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import params.param_loader as pl
import modules.module_loader as ml

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="Path to the parameter file")

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = pl.load_param_file(param_file)

# Load dataset
params.data_shape = [28, 28, 1]
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='../Datasets/', train=True, download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.0,), std=(255,))])),
    batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=False)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='../Datasets/', train=False, download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.0,), std=(255,))])),
    batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=False)

# Load model
model = ml.load_model(params.model_type)
model.setup(params).to(params.device)

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
        if(params.model_type == "mlp"):
          output = model(data) # forward pass
          loss = model.loss(dict(zip(["prediction", "target"], [output, target])))
        else:
          recon, latents = model(data) # forward pass
          loss_dict = dict(zip(["reconstruction", "latents"], [recon, latents]))
          loss = model.loss(loss_dict)
        loss.backward() # backward pass
        optimizer.step()
        if(params.model_type == "lca"):
            with torch.no_grad():
                model.w.div_(torch.norm(model.w, dim=0, keepdim=True))
        if(params.model_type == "mlp"):
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if(batch_idx % int(num_batches/params.train_logs_per_epoch) == 0.):
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.1f}%".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    100. * correct / ((batch_idx+1) * len(data))
                    ))
        else:
            if(batch_idx % int(num_batches/params.train_logs_per_epoch) == 0.):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}\tRecon Loss: {:.6f}\tSparse Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    model.recon_loss.item(),
                    model.sparse_loss.item()
                    ))
    scheduler.step(epoch)


def test():
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
        print("\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.1f}%)\n"
              .format(test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))


# train model
for epoch in range(1, params.num_epochs+1):
    train(epoch, params)
    if(params.model_type == "mlp"):
        test()


PATH = model.cp_save_dir#"../Projects/"+params.model_type+"/savefiles/"
if not os.path.exists(PATH):
    os.makedirs(PATH)
SAVEFILE = PATH + "trained_model.pt"
torch.save(model.state_dict(), SAVEFILE)
