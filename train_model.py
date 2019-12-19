import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.mlp_model import MlpModel as mlp

from params.mlp_params import params as mlp_params
from params.lca_params import params as lca_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load params
params = mlp_params()

# Load dataset
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
model = mlp(params).to(device)

# Setup optimizer
if params.optimizer.type == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params.weight_lr,
        weight_decay=params.weight_decay)
elif params.optimizer.type == "adam":
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
    scheduler.step(epoch)
    model.train()
    epoch_size = len(train_loader.dataset)
    num_batches = epoch_size / params.batch_size
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clear gradietns of all optimized variables
        output = model(data) # forward pass
        loss = F.nll_loss(output, target)
        loss.backward() # backward pass
        optimizer.step()
        if(batch_idx % int(num_batches/params.train_logs_per_epoch) == 0.):
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
              .format(test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))

# train model
for epoch in range(1, params.num_epochs+1):
    train(epoch, params)
    test()
