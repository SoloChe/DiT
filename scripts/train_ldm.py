# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import sys
import os

sys.path.append(os.path.realpath("./"))

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse

import os

from models.models import DiT_models
from diffusion import create_diffusion

# from diffusers.models import AutoencoderKL
from utils import create_logger

from data_med import BrainDataset_3D, BrainDataset_2D, normalise_percentile, get_age
from translation import sample_from_noise
from utils import load_from_checkpoint, load_from_checkpoint_vae
from generative.networks.nets import AutoencoderKL

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // (2)
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:

        os.makedirs(
            args.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))

        suffix = "-ldm-3D" if args.dim == 3 else "-lam-2D"
        model_string_name = (
            args.model.replace("/", "-") + suffix
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        args.img_dir = f"{experiment_dir}/images"  # Stores generated images
        os.makedirs(
            args.img_dir, exist_ok=True
        )  # Make image folder (holds all generated images)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger, writer = create_logger(experiment_dir)

        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"args: {args}")
    else:
        logger, writer = create_logger(None)

    prefix = None if args.prefix == "all" else args.prefix
    _, age_map, _ = get_age(args.age_path, prefix=prefix, round_age=True)  # age:index
    args.num_classes = len(age_map)

    latent_size = args.image_size
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        dim=args.dim,
        pos_embed_dim=args.pos_embed_dim,
    ).to(device)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # read model from checkpoint
    if args.resume_checkpoint is not None:
        # resume_steps, resume_epochs = load_from_checkpoint(model, ema, opt, args.resume_checkpoint)
        logger.info(f"Loading checkpoint from {args.resume_checkpoint}")
        model, ema, opt, resume_steps, scale_factor = load_from_checkpoint(
            model, ema, opt, args.resume_checkpoint, load_sf=True
        )
    else:
        resume_steps = 0
        scale_factor = None

    logger.info(f"set model to DDP...")
    model = DDP(model, device_ids=[rank])

    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule

    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64, 128),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, False, True),
    ).to(device)

    logger.info(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_from_checkpoint_vae(vae, args.vae_checkpoint)
    vae.eval()

    logger.info(
        f"DiT Parameters: {sum(p.numel() for p in model.parameters())/1024/1024:.2f}M"
    )
    logger.info(
        f"VAE Parameters: {sum(p.numel() for p in vae.parameters())/1024/1024:.2f}M"
    )

    if args.dim == 3:
        # dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train")
        dataset = BrainDataset_3D(
            args.data_path,
            args.age_path,
            prefix=prefix,
            mode="train",
            round_age=True,
            crop=True,
        )
    else:
        dataset = BrainDataset_2D(
            args.data_path,
            args.age_path,
            prefix=prefix,
            mode="train",
            round_age=True,
            crop=True,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(
        ema, model.module, decay=0
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = resume_steps
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for data in loader:

            x = data[0].to(device)
            y = data[1].to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode_stage_2_inputs(x).to(device)
                if train_steps == 0:
                    logger.info(f"Latent shape: {x.shape}")
                    scale_factor = 1.0 / (torch.std(x))

                x = x * scale_factor

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}, t.shape: {t.shape}")

            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # logger.info(f"loss: {loss.item()}")

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                writer.add_scalar("train/loss", avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scale_factor": scale_factor,
                        "args": args,
                        "resume_steps": train_steps,
                        "resume_epochs": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    logger.info("Sampling from noise...")

                    model.eval()
                    # logger.info(f"model training mode: {model.training}")
                    # logger.info(f"model module training mode : {model.module.training}", )
                    _, _ = sample_from_noise(
                        model.module,
                        diffusion,
                        device,
                        args,
                        name=train_steps,
                        decoder=vae.decode_stage_2_outputs,
                        scale_factor=scale_factor,
                    )
                    model.train()

                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--age-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument(
        "--image-size", type=int, choices=[28, 32, 128, 224, 256, 512], default=256
    )
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument(
        "--pos-embed-dim", type=int, default=3
    )  # posional embedding: 1d:1 2d:2 3d:3 or learned:4
    parser.add_argument("--prefix", type=str, default="IXI")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)

    parser.add_argument("--cfg-scale", type=float, default=1.8)
    parser.add_argument(
        "--labels",
        type=float,
        nargs="+",
        help="labels (index) to sample from periodically during training",
        default=10,  # disabled in default
    )

    args = parser.parse_args()
    main(args)
