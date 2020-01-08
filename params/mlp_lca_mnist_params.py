import os
import types
import numpy as np
import torch

from params.base_params import BaseParams
from params.lca_mnist_params import params as LcaParams
from params.mlp_mnist_params import params as MlpParams

class params(BaseParams):
    class mlp_params(BaseParams):
        def set_params(self):
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
            MlpParams.compute_helper_params(self)

    class lca_params(BaseParams):
        def set_params(self):
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
            self.renormalize_weights = True
            self.dt = 0.001
            self.tau = 0.03
            self.num_steps = 75
            self.rectify_a = True
            self.thresh_type = "soft"
            self.sparse_mult = 0.3
            self.weight_lr = 0.1
            self.num_latent = (28*28)*4

        def compute_helper_params(self):
            LcaParams.compute_helper_params(self)

    def set_params(self):
        super(params, self).set_params()
        self.ensemble_params = [mlp_params(), lca_params()]
