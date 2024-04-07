import argparse
import os
import torch 
import torch.distributed as dist

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)  

def load_from_checkpoint(model, ema, opt, model_name):
    """
    load states from a checkpoint
    """
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    opt.load_state_dict(checkpoint["opt"])
    
    # sync_params(model.parameters())
    # sync_params(ema.parameters())
    
    steps = checkpoint["resume_steps"]
    return steps
    
    
    
  

