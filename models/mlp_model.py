import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpModel(nn.Module):
    def __init__(self, dropout_rate):
        super(MlpModel, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=3136, bias=True)
        self.fc2 = nn.Linear(in_features=3136, out_features=10, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
