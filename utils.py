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

def load_from_checkpoint(model, ema, opt, checkpoint, device):
    """
    load states from a checkpoint
    """
    assert os.path.isfile(checkpoint), f'Could not find DiT checkpoint at {checkpoint}'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
    
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    opt.load_state_dict(checkpoint["opt"])
    
    # sync_params(model.parameters())
    # sync_params(ema.parameters())
    
    steps = checkpoint["resume_steps"]
    return model, ema, opt, steps
    
    
    
  

