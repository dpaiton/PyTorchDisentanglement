import os
import types
import numpy as np
import torch

class BaseParams(object):
    def __init__(self):
        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
