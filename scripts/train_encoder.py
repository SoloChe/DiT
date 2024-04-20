
import sys
import os
sys.path.append(os.path.realpath('./'))

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

from monai import transforms
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm


from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from data_med import BrainDataset_3D
from utils import create_logger

import argparse
from pynvml import *


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def main(args):
        # TODO: Add DDP
        # assert torch.cuda.is_available(), "Training with DDP requires at least one GPU."
        
        # # Set up DDP:
        # dist.init_process_group("nccl")
        # assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by the number of GPUs."
        
        # rank = dist.get_rank()
        # device = rank % torch.cuda.device_count()
        # seed = args.global_seed * dist.get_world_size() + rank
        # torch.manual_seed(seed)
        # torch.cuda.set_device(device)
        # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        
        
        # Set up logging:
        # if rank == 0:
        #     os.makedirs(args.log_path, exist_ok=True)
        #     os.mkdedirs(args.img_path, exist_ok=True)
        #     logger, writer = create_logger(args.log_path)
        #     logger.info(f'Experiment directory created at {args.log_dir}')
        #     logger.info(args)
        # else:
        #     logger, writer = create_logger(None)
        
        # Setup logs and device
        assert torch.cuda.is_available(), "Training with DDP requires at least one GPU."
        device = torch.device("cuda")
        
        # Get handle for the first GPU device
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        # Get memory info
        info = nvmlDeviceGetMemoryInfo(handle)
        
        os.makedirs(args.log_path, exist_ok=True)
        
        checkpoints_path = os.path.join(args.log_path, "checkpoints")
        image_path = os.path.join(args.log_path, "images")
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        
        logger, writer = create_logger(args.log_path, ddp=False)
        logger.info(f'Experiment directory created at {args.log_path}')
        logger.info(args)

        # Set up dataset:
        train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train")
        val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,    
            num_workers=args.num_workers,
        )
    
        logger.info(f"Dataset contains {len(train_dataset):,} images")
        logger.info(f"Validation dataset contains {len(val_dataset):,} images")
    
        # Set up encoder:
        autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64, 128),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, False, True),
        )
        autoencoder.to(device)
        
        discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
        discriminator.to(device)
        
        l1_loss = L1Loss()
        adv_loss = PatchAdversarialLoss(criterion="least_squares")
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        loss_perceptual.to(device)
        
        adv_weight = 0.01
        perceptual_weight = 0.001
        kl_weight = 1e-6

        optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)
        
        logger.info(f"Autoencoder Model with {sum(p.numel() for p in autoencoder.parameters()) / 1024 / 1024:.2f}M parameters.")
        logger.info(f"Discriminator Model with {sum(p.numel() for p in discriminator.parameters()) / 1024 / 1024:.2f}M parameters.")
       
        # Training loop
        train_steps = 0
        for epoch in range(args.epochs):
            autoencoder.train()
            discriminator.train()
            loss = 0
            gen_loss = 0
            disc_loss = 0
           
            for step, data in enumerate(train_loader):
                images = data[0].to(device)  # [B, C, H, W, D]
                 
                train_steps += 1
                # Generator part
                optimizer_g.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                
                print(reconstruction.shape, z_mu.shape, z_sigma.shape)
                
                kl_loss = KL_loss(z_mu, z_sigma)

                recons_loss = l1_loss(reconstruction.float(), images.float())
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

                if epoch > args.autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss

                loss_g.backward()
                optimizer_g.step()

                if epoch > args.autoencoder_warm_up_n_epochs:
                    # Discriminator part
                    optimizer_d.zero_grad(set_to_none=True)
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = adv_weight * discriminator_loss

                    loss_d.backward()
                    optimizer_d.step()

                loss += recons_loss.item()
                if epoch > args.autoencoder_warm_up_n_epochs:
                    gen_loss += generator_loss.item()
                    disc_loss += discriminator_loss.item()

                    if train_steps % args.log_steps == 0:
                        logger.info(f"recons_loss: {loss / (step + 1):.4f}, gen_loss: {gen_loss / (step + 1):.4f}, disc_loss: {disc_loss / (step + 1):.4f}")
                        writer.add_scalar("train/recons_loss", loss / (step + 1), train_steps)
                        
                    if train_steps % args.save_steps == 0:
                        checkpoints = {'autoencoder': autoencoder.state_dict(), 
                                    'discriminator': discriminator.state_dict(),
                                    'optimizer_g': optimizer_g.state_dict(),
                                    'optimizer_d': optimizer_d.state_dict(),
                                    'train_steps': train_steps,
                                    'args': args}
                        
                        torch.save(checkpoints, os.path.join(checkpoints_path, f"checkpoint_{train_steps}.pt"))
                        
                        img_plot = images.detach().cpu() # [B, C, H, W, D]
                        img_plot = img_plot[0, :, :, :, 112-10:112+10] # [1, H, W, 20] # visual check 20 slices (top view)
                        img_plot = torch.einsum('chwd->dchw', img_plot) # [20, 1, H, W]
                        
                        recon_imgs = reconstruction.detach().cpu() # [B, C, H, W, D]
                        recon_imgs = recon_imgs[0, :, :, :, 112-10:112+10] # [1, H, W, 20] # visual check 20 slices (top view)
                        recon_imgs = torch.einsum('chwd->dchw', recon_imgs) # [20, 1, H, W]
                        
                        plots = torch.cat([img_plot, recon_imgs], dim=0)
                        
                        plots = make_grid(plots, nrow=5)
                        save_image(plots, os.path.join(image_path, f"recon_{train_steps}.png"))
                        
                        
                        
                        
                        
                        
                        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./logs_vae")
    parser.add_argument("--data_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/extracted/")
    parser.add_argument("--age_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--autoencoder_warm_up_n_epochs", type=int, default=5)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    
    
    args = parser.parse_args()
    main(args)
        