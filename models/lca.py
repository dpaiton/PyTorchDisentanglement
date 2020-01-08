import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from modules.activations import lca_threshold

class Lca(BaseModel):
    def setup_model(self):
        self.w = nn.Parameter(torch.randn(self.params.num_pixels, self.params.num_latent),
            requires_grad=True)

    def preprocess_data(self, input_tensor):
        input_tensor = input_tensor.view(-1, self.params.num_pixels)
        return input_tensor

    def compute_excitatory_current(self, input_tensor):
        return torch.matmul(input_tensor, self.w)

    def compute_inhibitory_connectivity(self):
        lca_g = torch.matmul(torch.transpose(self.w, dim0=0, dim1=1),
            self.w) - torch.eye(self.params.num_latent,
            requires_grad=True, device=self.params.device)
        return lca_g

    def threshold_units(self, u_in):
        a_out = lca_threshold(u_in, self.params.thresh_type, self.params.rectify_a,
             self.params.sparse_mult)
        return a_out

    def step_inference(self, u_in, a_in, b, g, step):
        lca_explain_away = torch.matmul(a_in, g)
        du = b - lca_explain_away - u_in
        u_out = u_in + self.params.step_size * du
        return u_out, lca_explain_away

    def infer_coefficients(self, input_tensor):
        lca_b = self.compute_excitatory_current(input_tensor)
        lca_g = self.compute_inhibitory_connectivity()
        u_list = [torch.zeros([input_tensor.shape[0], self.params.num_latent],
          device=self.params.device)]
        a_list = [self.threshold_units(u_list[0])]
        for step in range(self.params.num_steps-1):
            u = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)[0]
            u_list.append(u)
            a_list.append(self.threshold_units(u))
        return (u_list, a_list)

    def get_recon_from_latents(self, latents):
        return torch.matmul(latents, torch.transpose(self.w, dim0=0, dim1=1))

    def get_encodings(self, input_tensor):
        u_list, a_list = self.infer_coefficients(input_tensor)
        return a_list[-1]

    def forward(self, input_tensor):
        input_tensor = self.preprocess_data(input_tensor)
        latents = self.get_encodings(input_tensor)
        reconstruction = self.get_recon_from_latents(latents)
        return reconstruction

    def get_recon_loss(self, input_tensor, recon):
        data_reduc_dim = list(range(1, len(recon.shape)))
        mse = torch.pow(input_tensor - recon, 2.)
        recon_loss = torch.mean(torch.sum(0.5 * mse, dim=data_reduc_dim, keepdim=False))
        return recon_loss

    def get_sparse_loss(self, latents):
        latent_reduc_dim = list(range(1, len(latents.shape)))
        sparse_loss = self.params.sparse_mult * torch.mean(torch.sum(torch.abs(latents),
            dim=latent_reduc_dim, keepdim=False))
        return sparse_loss

    def get_total_loss(self, input_tuple):
        input_tensor, input_labels = input_tuple
        input_tensor = self.preprocess_data(input_tensor)
        latents = self.get_encodings(input_tensor)
        recon = self.get_recon_from_latents(latents)
        recon_loss = self.get_recon_loss(input_tensor, recon)
        sparse_loss = self.get_sparse_loss(latents)
        total_loss = recon_loss + sparse_loss
        return total_loss

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(Lca, self).generate_update_dict(input_data, input_labels, batch_step)
        epoch = batch_step / self.params.batches_per_epoch
        stat_dict = {
            "epoch":int(epoch),
            "batch_step":batch_step,
            "train_progress":np.round(batch_step/self.params.num_batches, 3)}
        input_data = self.preprocess_data(input_data)
        latents = self.get_encodings(input_data)
        recon = self.get_recon_from_latents(latents)
        recon_loss = self.get_recon_loss(input_data, recon).item()
        sparse_loss = self.get_sparse_loss(latents).item()
        stat_dict["recon_loss"] = recon_loss
        stat_dict["sparse_loss"] = sparse_loss
        stat_dict["total_loss"] = recon_loss + sparse_loss
        update_dict.update(stat_dict)
        return update_dict
