import os
import types
import numpy as np
import torch

from params.base_params import BaseParams

class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = "lca"
        self.model_name = "lca_mnist"
        self.version = "0.0"
        self.dataset = "mnist"
        self.batch_size = 50
        self.num_epochs = 50
        self.weight_lr = 1e-4
        self.weight_decay = 0.
        self.train_logs_per_epoch = 3
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = "sgd"
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.1

        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = "soft"
        self.sparse_mult = 0.3
        self.weight_lr = 0.1
        self.num_latent = (28*28)*4
        self.compute_helper_params()

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.step_size = self.dt / self.tau
