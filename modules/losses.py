import numpy as np
import torch

def half_squared_l2(x1, x2):
    """
    Computes the standard reconstruction loss. It will average over batch dimensions.
    Args:
        original: Tensor with original input image
        reconstruction: Tensor with reconstructed image for comparison
    Returns:
        recon_loss: Tensor representing the squared l2 distance between the inputs, averaged over batch
    """
    raise NotImplementedError
