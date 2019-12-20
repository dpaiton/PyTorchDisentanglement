import os
import types
import numpy as np
import torch

from params.base_params import BaseParams

class params(BaseParams):
    def __init__(self):
        super(params, self).__init__()
        self.batch_size = 50
        self.num_epochs = 50
        self.weight_lr = 1e-4
        self.weight_decay = 5e-4
        self.dropout_rate = 0.6 # probability of value being set to zero
        self.train_logs_per_epoch = 3
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = "adam"
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.9
        self.num_latent = (28*28)*6

        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = "soft"
        self.sparse_mult = 0.3
        self.weight_lr = 0.1
        self.optimizer_type = "sgd"
        self.num_latent = (28*28)*4
        self.compute_helper_params()

    def compute_helper_params(self):
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_size = self.dt / self.tau
