# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import sys
import os

sys.path.append(os.path.realpath("./"))

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image, make_grid
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models.models import DiT_models
import argparse
from data_med import get_age, BrainDataset_3D, BrainDataset_2D
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from generative.networks.nets import AutoencoderKL
from models.U_DiT import U_DiT

from utils import (
    str2bool,
    create_logger,
    load_from_checkpoint_vae,
    load_from_checkpoint,
)


def sample_each_age_3D(
    model, diffusion, source, age_map, device, args, vae, scale_factor=None
):
    assert (vae is not None) == (
        scale_factor is not None
    ), "Decoder and scale_factor must be provided for LDM"
    SAMPLES = []
    index = list(age_map.values())

    for age_index in index:  # age:index

        y = (
            (
                torch.ones(source.shape[0], dtype=torch.long)
                * (torch.tensor(age_index).reshape(-1, 1))
            )
            .flatten()
            .to(device)
        )

        t = torch.tensor([args.num_noise_steps - 1] * source.shape[0], device=device)

        latent = vae.encode_stage_2_inputs(source)

        z = diffusion.q_sample(latent, t=t, noise=None)
        z = torch.cat([z, z], 0)

        y_null = torch.tensor([args.num_classes] * latent.shape[0], device=device)
        y = torch.cat([y, y_null], 0)

        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            noise=z,
            noise_steps=args.num_noise_steps,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )  # n, latent_channel, latent_size, latent_size, latent_size

        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = (
            vae.decode_stage_2_outputs(samples / scale_factor)
            if vae is not None
            else samples
        )  # batch_size, 1 224 224 224
        samples = samples.reshape(1, *source.shape).cpu()
        SAMPLES.append(samples)

    SAMPLES = torch.cat(SAMPLES, dim=0)  # len(age_map), batch_size, 1, 224, 224, 224
    return SAMPLES, z.chunk(2, dim=0)[0]


def sample_each_age(model, diffusion, source, age_map, device, args):

    # assert (decoder is not None) == (scale_factor is not None), "Decoder and scale_factor must be provided for LDM"

    SAMPLES = []
    index = list(age_map.values())
    n = 4
    index_n = (
        [index[i : i + n] for i in range(0, len(index), n)] if n > 1 else index
    )  # n ages per time

    noise = torch.randn_like(source)

    for age_index in index_n:  # age:index

        y = (
            (
                torch.ones(source.shape[0], dtype=torch.long)
                * (torch.tensor(age_index).reshape(-1, 1))
            )
            .flatten()
            .to(device)
        )

        source_n = source.repeat(
            len(age_index), 1, 1, 1
        )  # source batch_size, 1, 224, 224 -> batch_size*n, 1, 224, 224

        noise_n = noise.repeat(len(age_index), 1, 1, 1)

        t = torch.tensor([args.num_noise_steps - 1] * source_n.shape[0], device=device)

        # TODO should make sure the noise is the same for the all ages
        z = diffusion.q_sample(source_n, t=t, noise=noise_n)
        z = torch.cat([z, z], 0)

        y_null = torch.tensor([args.num_classes] * source_n.shape[0], device=device)
        y = torch.cat([y, y_null], 0)

        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            noise=z,
            noise_steps=args.num_noise_steps,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )

        # move to cpu
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = samples.reshape(
            len(age_index), 224, 1, 224, 224
        ).cpu()  # batch_size*n, 1, 224, 224 -> n, batch_size, 1, 224, 224
        SAMPLES.append(samples)

    SAMPLES = torch.cat(
        SAMPLES, dim=0
    )  # (len(age_map), batch_size, 1, 224, 224, 224) or (len(age_map), batch_size, 1, 224, 224)
    return SAMPLES, z.chunk(2, dim=0)[0]


def sample_from_noise(
    model, diffusion, device, args, name=None, decoder=None, scale_factor=None
):

    assert (decoder is not None) == (
        scale_factor is not None
    ), "Decoder and scale_factor must be provided for LDM"

    y = args.labels
    assert isinstance(y, list)
    y = torch.tensor(y, dtype=torch.long).to(device)

    if args.dim == 3:
        z = torch.randn(
            len(y),
            args.in_channels,
            args.image_size,
            args.image_size,
            args.image_size,
            device=device,
        )
    else:
        z = torch.randn(
            len(y), args.in_channels, args.image_size, args.image_size, device=device
        )

    y_null = torch.tensor([args.num_classes] * z.shape[0], device=device)
    y = torch.cat([y, y_null], 0)
    z = torch.cat([z, z], 0)

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        noise=z,
        noise_steps=None,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )  # (2 * len(y), 1, 224, 224, 224) or (2 * len(y), 1, 224, 224) or (2*len(y), patch_size, patch_size, patch_size)

    samples, _ = samples.chunk(
        2, dim=0
    )  # Remove null class samples (len(y), 1, 224, 224, 224) or (len(y), 1, 224, 224)

    samples = decoder(samples / scale_factor) if decoder is not None else samples

    if name is not None:
        if args.dim == 3:
            samples = samples.squeeze(
                1
            )  # 1, 224, 224, 224 or 1 path_size, path_size, path_size
            samples = torch.einsum(
                "chwd->dchw", samples
            )  # 224, 1, 224, 224 or patch_size, 1, patch_size, patch_size
            # select middle 40 slice
            samples = samples[112 - 20 : 112 + 20, ...]

        samples = make_grid(samples, nrow=4)
        save_image(samples, Path(args.img_dir) / f"{name}.png")
    return samples, z.chunk(2, dim=0)[0]


def calculate_patient_mae(source, index, samples, start, end):
    """
    source: (batch_size, 1, 224, 224) batch_size = n_slices_per_patient = 224, or (batch_size, 1, 224, 224, 224) for 3D
    index: (batch_size,) age index
    samples: (len(age_map), batch_size, 1, 224, 224) or (len(age_map), batch_size, 1, 224, 224, 224)
    start: int start index of the slice
    end: int end index of the slice
    """
    source = source.cpu().unsqueeze(0)
    samples = samples.cpu()
    n_slices_per_patient = 224

    import scipy.stats as stats

    _, age_map, _ = get_age(
        "/data/amciilab/yiming/DATA/brain_age/masterdata.csv"
    )  # age:index
    reversed_age_map = {v: k for k, v in age_map.items()}  # index:age

    diff_map = (
        source - samples
    ) ** 2  # (len(age_map), batch_size, 1, 224, 224) or (len(age_map), batch_size, 1, 224, 224, 224)
    diff_map = (
        diff_map.mean(dim=(2, 3, 4))
        if len(diff_map.shape) == 5
        else diff_map.mean(dim=(2, 3, 4, 5))
    )  # len(age_map), batch_size
    predictions = diff_map.min(dim=0)[1]  # batch_size

    if len(source.shape) == 5:
        age_true_slice = np.array(
            [reversed_age_map[i.item()] for i in index]
        )  # batch_size
        age_pred_slice = np.array(
            [reversed_age_map[i.item()] for i in predictions]
        )  # batch_size

        age_true_slice = age_true_slice.reshape(
            -1, n_slices_per_patient
        )  # n_patients, n_slices_per_patient
        age_pred_slice = age_pred_slice.reshape(
            -1, n_slices_per_patient
        )  # n_patients, n_slices_per_patient

        age_pred_slice_partial = age_pred_slice[:, (112 - start) : (112 + end)]
        age_true_slice_partial = age_true_slice[:, (112 - start) : (112 + end)]

        age_pred_mod = stats.mode(age_pred_slice_partial, axis=1)[0]  # n_patients
        age_pred_mean = age_pred_slice_partial.mean(axis=1)  # n_patients

        age_true_patient = stats.mode(age_true_slice_partial, axis=1)[0]  # n_patients

        mae_mod = np.abs(age_true_patient - age_pred_mod)  # n_patients
        mae_mean = np.abs(age_true_patient - age_pred_mean)  # n_patients
    elif len(source.shape) == 6:
        age_true = np.array(
            [reversed_age_map[i.item()] for i in index]
        )  # batch_size: n_patient
        age_pred = np.array(
            [reversed_age_map[i.item()] for i in predictions]
        )  # batch_size: n_patient
        mae_mean = np.abs(age_true - age_pred)  # batch_size: n_patient
        mae_mod = mae_mean

    return mae_mod, mae_mean


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    logger, _ = create_logger(args.log_path, ddp=False, tb=False)
    logger.info(f"args: {args}")

    _, age_map, _ = get_age(
        args.age_path, round_age=True, prefix=args.prefix
    )  # age:index

    args.num_classes = len(age_map)

    if args.model == "UDiT":
        model = U_DiT(
            img_size=args.image_size,
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            spatial_dims=args.dim,
            pos_embed_dim=args.pos_embed_dim,
            learn_sigma=True,
        ).to(device)
    else:
        model = DiT_models[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            dim=args.dim,
            pos_embed_dim=args.pos_embed_dim,
        ).to(device)

    model, _, _, _ = load_from_checkpoint(
        model, checkpoint=args.DiT_checkpoint, load_sf=False
    )
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_total_steps))
    logger.info(f"Loaded DiT model from {args.DiT_checkpoint}")

    if args.from_noise:
        SAMPLES, noise = sample_from_noise(model, diffusion, device, args)

        SAMPLES_path = Path(args.log_path) / f"sample.npy"
        np.save(SAMPLES_path, SAMPLES.cpu().numpy())

        NOISE_path = Path(args.log_path) / f"noise.npy"
        np.save(NOISE_path, noise.cpu().numpy())

    else:
        prefix = None if args.prefix == "all" else args.prefix
        if args.dim == 3:
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

            data_test = BrainDataset_3D(
                args.data_path,
                args.age_path,
                mode="val",
                round_age=True,
                prefix=prefix,
                crop=True,
            )
        else:
            data_test = BrainDataset_2D(
                args.data_path,
                args.age_path,
                mode="val",
                round_age=True,
                prefix=prefix,
                crop=True,
            )

        loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

        running_mae_mod = [0 for _ in range(5, 101, 5)]
        running_mae_mean = [0 for _ in range(5, 101, 5)]

        for i, data in enumerate(loader):

            if i >= args.num_batches:
                break

            source, lab, id, age = data
            source = source.to(device)

            logger.info("-" * 80)
            logger.info(f"batch {i} is being translated...")

            # Translation
            if args.dim == 2:
                SAMPLES, noise = sample_each_age(
                    model, diffusion, source, age_map, device, args
                )
            else:
                SAMPLES, noise = sample_each_age_3D(
                    model,
                    diffusion,
                    source,
                    age_map,
                    device,
                    args,
                    vae=vae,
                    scale_factor=scale_factor,
                )

            logger.info(
                f"translation of batch {i} finished with sample shape: {SAMPLES.shape}"
            )

            # Age prediction
            if args.dim == 2:
                for j, num_slice in enumerate(range(5, 101, 5)):
                    mae_mode, mae_mean = calculate_patient_mae(
                        source, lab, SAMPLES, num_slice, num_slice
                    )

                    logger.info("+" * 50)
                    logger.info(f"MAE for batch {i} with middle {num_slice} slices:")

                    patient_name = list(set(id))[0]  # only 1 patient per batch
                    running_mae_mod[j] += mae_mode[0]
                    running_mae_mean[j] += mae_mean[0]

                    logger.info(f"patient: {patient_name}")
                    logger.info(f"mod {mae_mode[0]}, mean {mae_mean[0]}")
                    logger.info(
                        f"running MAE: mod {running_mae_mod[j]/(i+1)}, mean {running_mae_mean[j]/(i+1)}"
                    )
                    logger.info("+" * 50)
            else:
                mae_mode, mae_mean = calculate_patient_mae(source, lab, SAMPLES, 0, 0)
                logger.info(f"patient: {id[0]}")
                logger.info(f"MAE{mae_mode[0]}")
            logger.info("-" * 80)

            # Saving samples
            if args.save:
                logger.info(f"saving batch {i}...")

                SAMPLES_path = Path(args.log_path) / f"sample_{i}.npy"
                np.save(SAMPLES_path, SAMPLES.cpu().numpy())

                SOURCE_path = Path(args.log_path) / f"source_{i}.npy"
                np.save(SOURCE_path, source.cpu().numpy())

                NOISE_path = Path(args.log_path) / f"noise_{i}.npy"
                np.save(NOISE_path, noise.cpu().numpy())

                ID_path = Path(args.log_path) / f"id_{i}.npy"
                np.save(ID_path, id)

                LABEL_path = Path(args.log_path) / f"label_{i}.npy"
                np.save(LABEL_path, lab.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--data-path", type=str, default="./data/brainage")
    parser.add_argument("--age-path", type=str, default="./data/brainage/age.csv")
    parser.add_argument("--log-path", type=str, default="./logs/test")
    parser.add_argument("--DiT-checkpoint", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    # data
    parser.add_argument("--prefix", type=str, default="IXI")
    parser.add_argument(
        "--image-size", type=int, choices=[28, 32, 224, 224, 512], default=224
    )
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--from-noise", type=str2bool, default=False)
    # diffusion
    parser.add_argument("--num-total-steps", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.8)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--num-noise-steps", type=int, default=10)
    # DiT
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--pos-embed-dim", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)

    # others
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--labels",
        type=float,
        nargs="+",
        help="labels (index) to sample from",
        default=10,  # disabled in default
    )
    args = parser.parse_args()
    main(args)
