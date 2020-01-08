import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.base_utils as utils
import models.model_loader as ml

class Ensemble(BaseModel):
    def setup_model(self):
        self.models = []
        for model_idx, params in enumerate(self.params.ensemble_params):
            model = ml.load_model(params.model_type)
            model.setup(params)
            model.to(params.device)
            self.models.append(model)

    def forward(self, x):
        for model in models:
            x = model.get_encodings(x)
        return x

    def get_ensemble_losses(self, input_tuple):
        return [model.get_total_loss(input_tuple) for model in self.models]

    def get_total_loss(self, input_tuple):
        total_loss = self.get_ensemble_losses(input_tuple)
        return torch.stack(total_loss, dim=0).sum()
