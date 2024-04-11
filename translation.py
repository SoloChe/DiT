# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image, make_grid
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from data_med import get_age, BrainDataset_3D, BrainDataset_2D
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

from utils import str2bool


def sample_each_age(model, diffusion, source, age_map, device, args):
    SAMPLES = []
    for age_index in age_map.values():
        
        y = torch.ones(source.shape[0], dtype=torch.long) * age_index
        y = y.to(device)
        
        t = torch.tensor(
        [args.num_noise_steps - 1] * source.shape[0], device=device
        )

        z = diffusion.q_sample(source, t=t, noise=None)
        z = torch.cat([z, z], 0)
        
        y_null = torch.tensor([args.num_classes] * source.shape[0], device=device)
        y = torch.cat([y, y_null], 0)
        
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, noise=z, noise_steps=args.num_noise_steps,
            clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
        SAMPLES.append(samples.unsqueeze(0))
        
    SAMPLES = torch.cat(SAMPLES, dim=0)
    return SAMPLES, z.chunk(2, dim=0)[0]

def sample_from_noise(model, diffusion,  device, args, name=None):
    
    y = args.labels
    assert isinstance(y, list)
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    if args.dim == 3:
        z = torch.randn(len(y),
                     args.in_channels,
                     args.image_size,
                     args.image_size,
                     args.image_size,
                     device=device)
    else:
        z = torch.randn(len(y),
                        args.in_channels,
                        args.image_size,
                        args.image_size,
                        device=device)
    
    y_null = torch.tensor([args.num_classes] * z.shape[0], device=device)
    y = torch.cat([y, y_null], 0)
    
    z = torch.cat([z, z], 0)
    
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, noise=z, noise_steps=None,
        clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    ) # (2 * len(y), 1, 256, 256, 256) or (2 * len(y), 1, 256, 256) 
    
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples (len(y), 1, 256, 256, 256) or (len(y), 1, 256, 256)
    
    if name is not None:
        if args.dim == 3:
            samples = samples.squeeze(1) # 1, 256, 256, 256
            samples = torch.einsum("chwd->dchw", samples)  # 256, 1, 256, 256
            # select middle 20 slice
            samples = samples[118:138,...] # 20, 1, 256, 256
            
        samples = make_grid(samples, nrow=4)
        save_image(samples, Path(args.img_dir) / f"{name}.png")
    return samples, z.chunk(2, dim=0)[0]
                        
    
def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.log_path).mkdir(parents=True, exist_ok=True)
     
    
    latent_size = args.image_size
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        dim=args.dim,
        pos_embed_dim=args.pos_embed_dim,
    ).to(device)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt 
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_total_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    if args.from_noise:
        SAMPLES, noise = sample_from_noise(model, diffusion, device, args)
      
        SAMPLES_path = Path(args.log_path) / f"sample.npy"
        np.save(SAMPLES_path, SAMPLES.cpu().numpy())
        
        NOISE_path = Path(args.log_path) / f"noise.npy"
        np.save(NOISE_path, noise.cpu().numpy())
        
    else:
        _, age_map, _ = get_age(args.age_path) # age:index

        if args.dim == 3:
            data_test = BrainDataset_3D(args.data_path, args.age_path, mode="val") 
        else:
            data_test = BrainDataset_2D(args.data_path, args.age_path, mode="val")
        
        loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
        
        for i, data in enumerate(loader):
            
            if i >= args.num_batches:
                break
            
            source, lab, id = data
            source = source.to(device)
            SAMPLES, noise = sample_each_age(model, diffusion, source, age_map, device, args)
            
            SAMPLES_path = Path(args.log_path) / f"sample_{i}.npy"
            np.save(SAMPLES_path, SAMPLES.cpu().numpy())
            
            SOURCE_path = Path(args.log_path) / f"source_{i}.npy"
            np.save(SOURCE_path, source.cpu().numpy())
            
            NOISE_path = Path(args.log_path) / f"noise_{i}.npy"
            np.save(NOISE_path, noise.cpu().numpy())
            
            ID_path = Path(args.log_path) / f"id_{i}.npy"
            np.save(ID_path, id.numpy())
            
            LABEL_path = Path(args.log_path) / f"label_{i}.npy"
            np.save(LABEL_path, lab.numpy())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--data-path", type=str, default="./data/brainage")
    parser.add_argument("--age-path", type=str, default="./data/brainage/age.csv")
    parser.add_argument("--log-path", type=str, default="./logs/test")
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--from-noise", type=str2bool, default=False)
    
    parser.add_argument("--num-total-steps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, choices=[32, 224, 256, 512], default=256)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--pos-embed-dim", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.8)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--num-noise-steps", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument(
        "--labels",
        type=float,
        nargs="+",
        help="labels (index) to sample from",
        default=10,  # disabled in default
    )
    args = parser.parse_args()
    main(args)
