import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

class Mlp(BaseModel):
    def setup_model(self):
        self.fc1 = nn.Linear(
            in_features = self.params.num_pixels,
            out_features = self.params.num_latent,
            bias = True)
        self.fc2 = nn.Linear(
            in_features = self.params.num_latent,
            out_features = 10,
            bias = True)
        self.dropout = nn.Dropout(p=self.params.dropout_rate)

    def get_total_loss(self, input_tuple):
        input_tensor, input_label = input_tuple
        pred = self.forward(input_tensor)
        return F.nll_loss(pred, input_label)

    def preprocess_data(self, input_tensor):
        input_tensor = input_tensor.view(-1, self.params.num_pixels)
        return input_tensor

    def forward(self, x):
        x = self.preprocess_data(x)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
    def get_encodings(self, x):
        return self.forward(self, x)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(Mlp, self).generate_update_dict(input_data, input_labels, batch_step)
        epoch = batch_step / self.params.batches_per_epoch
        stat_dict = {
            "epoch":int(epoch),
            "batch_step":batch_step,
            "train_progress":np.round(batch_step/self.params.num_batches, 3)}
        output = self.forward(input_data)
        pred = output.max(1, keepdim=True)[1]
        total_loss = self.get_total_loss((input_data, input_labels))
        correct = pred.eq(input_labels.view_as(pred)).sum().item()
        stat_dict["loss"] = total_loss.item()
        stat_dict["train_accuracy"] = 100. * correct / self.params.batch_size
        update_dict.update(stat_dict)
        return update_dict
