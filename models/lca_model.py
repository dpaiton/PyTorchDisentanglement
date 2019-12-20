import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.parameter import Parameter

from modules.activations import lca_threshold

class LcaModel(nn.Module):
    def __init__(self, params):
        super(LcaModel, self).__init__()
        self.params = params
        self.params.input_size = 28*28

        self.w = nn.Parameter(torch.randn(self.params.input_size, self.params.num_latent))
        self.w.requirs_grad = True

    def compute_excitatory_current(self):
        return torch.matmul(self.data_tensor, self.w)

    def compute_inhibitory_connectivity(self):
        lca_g = torch.matmul(torch.transpose(self.w, dim0=0, dim1=1),
            self.w) - torch.eye(self.params.num_latent,
            requires_grad=True)
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

    def infer_coefficients(self):
        lca_b = self.compute_excitatory_current()
        lca_g = self.compute_inhibitory_connectivity()
        u_list = [torch.zeros([self.data_tensor.shape[0], self.params.num_latent])]
        a_list = [self.threshold_units(u_list[0])]
        for step in range(self.params.num_steps-1):
            u = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)[0]
            u_list.append(u)
            a_list.append(self.threshold_units(u))
        return (u_list, a_list)

    def build_decoder(self, input_tensor):
        return torch.matmul(input_tensor, torch.transpose(self.w, dim0=0, dim1=1))

    def loss(self, dictargs):
        data_reduc_dim = list(range(1, len(self.data_tensor.shape)))
        latent_reduc_dim = list(range(1, len(dictargs["latents"].shape)))
        mse = torch.pow(self.data_tensor - dictargs["reconstruction"], 2.)
        recon_loss = torch.mean(torch.sum(0.5 * mse, dim=data_reduc_dim, keepdim=False))
        sparse_loss = torch.mean(torch.sum(torch.abs(dictargs["latents"]),
            dim=latent_reduc_dim, keepdim=False))
        return recon_loss + self.params.sparse_mult * sparse_loss

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        self.data_tensor = x
        u_list, a_list = self.infer_coefficients()
        self.u = u_list[-1]
        self.a = a_list[-1]
        self.reconstruction = self.build_decoder(self.a)
        return self.reconstruction, self.a
