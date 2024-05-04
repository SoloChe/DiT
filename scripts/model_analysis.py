
import os
import sys
sys.path.append(os.path.realpath("./"))

import torch
import torch.nn as nn

from models import U_DiT
from utils import create_logger
import numpy as np

from data_med import BrainDataset_3D, get_age, normalise_percentile

def main(args):
    
     # Set up logging:
    os.makedirs(args.logging_dir, exist_ok=True)
    logger, _ = create_logger(args.logging_dir, ddp=False, tb=False)
    logger.info(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefix = None if args.prefix == "all" else args.prefix
    crop = args.crop
    transform = normalise_percentile if args.use_trans else None
    
    val_dataset = BrainDataset_3D(
            args.data_path,
            args.age_path,
            mode="val",
            transform=transform,
            prefix=prefix,
            round_age=True,
            crop=crop,
            oversample=False,
        )
    
    model = U_DiT(
        img_size=args.image_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        spatial_dims=args.dim,
        pos_embed_dim=args.pos_embed_dim,
        learn_sigma=True,
    ).to(device)
    
    model.eval()
    
    #TODO Should check the attention map and CKA similarity?