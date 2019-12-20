import torch
import torch.nn as nn
import torch.nn.functional as F

def lca_threshold(u_in, thresh_type, rectify, sparse_threshold):
    if thresh_type == "soft":
        if rectify:
            a_out = torch.where(torch.gt(u_in, sparse_threshold), u_in - sparse_threshold,
                torch.zerors_like(u_in))
        else:
            a_out = torch.where(torch.ge(u_in, sparse_threshold), u_in - sparse_threshold,
                torch.where(torch.le(u_in, -sparse_threshold), u_in + sparse_threshold,
                tf.zeros_like(u_in)))
    elif thresh_type == "hard":
        if rectify:
            a_out = torch.where(torch.gt(u_in, sparse_threshold), u_in, torch.zeros_like(u_in))
        else:
            a_out = torch.where(torch.ge(u_in, sparse_threshold), u_in,
                torch.where(torch.le(u_in, -sparse_threshold), u_in, torch.zeros_like(u_in)))
    else:
        assert False, ("Parameter thresh_type must be 'soft' or 'hard', not "+thresh_type)
    return a_out
