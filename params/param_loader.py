import utils.base_utils as utils

def load_param_file(file_name):
    params_module = utils.module_from_file("params", file_name)
    return params_module.params()

if __name__ == "__main__":
    # Load params from file, parse it, print it out
    print("TODO.")
