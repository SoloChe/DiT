import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    assert os.path.isfile(checkpoint), f"Could not find DiT checkpoint at {checkpoint}"
    checkpoint = torch.load(checkpoint, map_location=torch.device("cuda"))

    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    opt.load_state_dict(checkpoint["opt"])

    # sync_params(model.parameters())
    # sync_params(ema.parameters())

    steps = checkpoint["resume_steps"]
    return model, ema, opt, steps


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass
        # Implement other methods as needed

def setup_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )

def create_logger(logging_dir, ddp=True, tb=True):
    """
    Create a logger that writes to a log file and stdout.
    """
    if ddp:
        if dist.get_rank() == 0:  # real logger
            setup_logger(logging_dir)
            writer = SummaryWriter(logging_dir) if tb else None
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
            writer = DummyWriter() if tb else None
    else:
        setup_logger(logging_dir)
        writer = SummaryWriter(logging_dir) if tb else None
        logger = logging.getLogger(__name__)
    return logger, writer
