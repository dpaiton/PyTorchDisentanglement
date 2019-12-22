import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger

class BaseModule(nn.Module):
    def __init__(self):
        self.params_loaded = False

    def setup(self, params):
        self.load_params(params)
        #self.check_params()
        self.make_dirs()
        self.init_logging()
        self.log_params()
        self.setup_model()

    def load_params(self, params):
        """
        Calculates a few extra parameters
        Sets parameters as member variable
        """
        params.cp_latest_filename = "latest_checkpoint_v"+params.version
        if not hasattr(params, "model_out_dir"):
            params.model_out_dir = params.out_dir + params.model_name
        params.cp_save_dir = params.model_out_dir + "/checkpoints/"
        params.log_dir = params.model_out_dir + "/logfiles/"
        params.save_dir = params.model_out_dir + "/savefiles/"
        params.disp_dir = params.model_out_dir + "/vis/"
        params.num_pixels = int(np.prod(params.data_shape))
        self.params = params
        self.params_loaded = True

    def get_param(self, param_name):
        """
        Get param value from model
          This is equivalent to self.param_name, except that it will return None if
          the param does not exist.
        """
        if hasattr(self, param_name):
            return getattr(self, param_name)
        else:
            return None

    def make_dirs(self):
        """Make output directories"""
        if not os.path.exists(self.params.model_out_dir):
            os.makedirs(self.params.model_out_dir)
        if not os.path.exists(self.params.log_dir):
            os.makedirs(self.params.log_dir)
        if not os.path.exists(self.params.cp_save_dir):
            os.makedirs(self.params.cp_save_dir)
        if not os.path.exists(self.params.save_dir):
            os.makedirs(self.params.save_dir)
        if not os.path.exists(self.params.disp_dir):
            os.makedirs(self.params.disp_dir)

    def init_logging(self, log_filename=None):
        if self.params.log_to_file:
            if log_filename is None:
                log_filename = self.params.log_dir+self.params.model_name+"_v"+self.params.version+".log"
                self.logger = Logger(filename=log_filename, overwrite=True)
        else:
            self.logger = Logger(filename=None)

    def js_dumpstring(self, obj):
        """Dump json string with special NumpyEncoder"""
        return self.logger.js_dumpstring(obj)

    def log_params(self, params=None):
        """Use logging to write model params"""
        if params is not None:
            dump_obj = params.__dict__
        else:
            dump_obj = self.params.__dict__
        self.logger.log_params(dump_obj)
