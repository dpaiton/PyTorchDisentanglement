import os
import types
import numpy as np
import torch

from params.base_params import BaseParams

class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = "mlp"
        self.model_name = "mlp_mnist"
        self.version = "0.0"
        self.dataset = "mnist"
        self.batch_size = 50
        self.num_epochs = 100
        self.weight_lr = 5e-4
        self.weight_decay = 2e-6
        self.dropout_rate = 0.5# 0.4 # probability of value being set to zero
        self.train_logs_per_epoch = 4
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = "adam"
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.1
        self.num_latent = (28*28)*4#(28*28)*6

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
