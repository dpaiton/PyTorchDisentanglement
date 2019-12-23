import utils.base_utils as utils

def load_model(model_type):
    if(model_type == "mlp"):
        module_name = "Mlp"
        file_name = "models/mlp.py"

    elif(model_type == "lca"):
        module_name = "Lca"
        file_name = "models/lca.py"

    else:
        assert False, ("Acceptible model_types are 'mlp' and 'lca'")

    module = utils.module_from_file(module_name, file_name)
    model = getattr(module, module_name)
    return model()

if __name__ == "__main__":
    # print list of models
    print("TODO.")
