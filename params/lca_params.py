import os
from params.base_params import BaseParams

class params(BaseParams):
    def __init__(self):
        super(params, self).__init__()
        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = "soft"
        self.sparse_mult = 0.3
        self.weight_lr = 0.1
        self.optimizer_type = "sgd"
