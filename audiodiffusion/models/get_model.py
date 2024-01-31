import importlib

def get_model(cfg, datamodule):
    modeltype = cfg.model.model_type
    return get_module(cfg, datamodule)
    # if modeltype in ["proto","faceformer"]:
    #     return get_module(cfg, datamodule)
    # else:
    #     raise ValueError(f"Invalid model {modeltype}.")
    
def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="alm.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    return Model(cfg=cfg, datamodule=datamodule)
 
    