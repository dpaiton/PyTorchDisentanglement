import os
import types
import numpy as np

class BaseParams(object):
    def __init__(self):
        self.batch_size = 40
        self.num_epochs = 50
        self.weight_lr = 1e-4
        self.weight_decay = 1e-5
        self.dropout_rate = 0.3 # probability of value being set to zero
        self.train_logs_per_epoch = 5
        self.optimizer = types.SimpleNamespace()
        self.optimizer.type = "adam"
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.9
        self.num_latent = 3136

        self.compute_helper_params()

    def compute_helper_params(self):
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


