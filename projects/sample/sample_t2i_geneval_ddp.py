# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import json
import torch.distributed as dist
from diffusers.models import AutoencoderKL, AutoencoderDC
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import functools
import argparse
from omegaconf import OmegaConf
from einops import rearrange
from tim.schedulers.transition import TransitionSchedule
from tim.utils.misc_utils import instantiate_from_config, init_from_ckpt
from tim.models.vae import (
    get_sd_vae, get_dc_ae,
    sd_vae_decode, dc_ae_decode
)
from tim.models.utils.text_encoders import load_text_encoder, encode_prompt
from safetensors.torch import load_file, save_file

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup dtype
    dtype = torch.bfloat16

    # Load model:
    config = OmegaConf.load(args.config)
    model_config = config.model 
    
    if 'dc-ae' in model_config.vae_dir:
        dc_ae = get_dc_ae(model_config.vae_dir, dtype=torch.float32, device=device)
        spatial_downsample = 32
        dc_ae.enable_tiling(2560, 2560, 2560, 2560)
        decode_func = functools.partial(dc_ae_decode, dc_ae, slice_vae=args.slice_vae)
    elif 'sd-vae' in model_config.vae_dir:
        sd_vae = get_sd_vae(model_config.vae_dir, dtype=torch.float32, device=device)
        spatial_downsample = 8
        decode_func = functools.partial(sd_vae_decode, sd_vae, slice_vae=args.slice_vae)
    else: raise
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    # image resolution
    latent_h = int(args.height / spatial_downsample)
    latent_w = int(args.width / spatial_downsample)

    # load text-encoder
    text_encoder, tokenizer = load_text_encoder(
        text_encoder_dir=model_config.text_encoder_dir, device=device, weight_dtype=torch.bfloat16
    )
    null_cap_feat, null_cap_mask = encode_prompt(
        tokenizer, text_encoder, device, torch.bfloat16, 
        [""], model_config.use_last_hidden_state, 
        max_seq_length=model_config.max_seq_length
    )
    
    model = instantiate_from_config(model_config.network).to(device=device, dtype=dtype)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=True)
    model.eval()  # important!
    
    transport = instantiate_from_config(model_config.transport)
    scheduler = TransitionSchedule(
        transport=transport, **OmegaConf.to_container(model_config.transition_loss)
    )
    
        
    
    # Create folder to save samples:
    model_name = args.ckpt.split('/')[-1].split('.')[0]
    folder_name = f"{args.data_type}-{model_name}-{args.height}x{args.width}-T{args.T_min}-{args.T_max}-" \
                  f"Step-{args.num_steps}-sto-{args.stochasticity_ratio}-{args.sample_type}-" \
                  f"cfg-{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    with open(args.caption_dir) as fp:
        all_metadatas = [json.loads(line) for line in fp]
    all_captions = [metadata['prompt'].encode('unicode-escape').decode('utf-8') for metadata in all_metadatas]
    all_paths = [os.path.join(sample_folder_dir, f"{i:0>5}") for i in range(len(all_metadatas))]
    total_samples = int(math.ceil(len(all_metadatas) / global_batch_size) * global_batch_size)
    pad_num = total_samples - len(all_metadatas)
    all_metadatas.extend(all_metadatas[:pad_num])
    all_captions.extend(all_captions[:pad_num])
    all_paths.extend(all_paths[:pad_num])

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_per_gpu = int(total_samples // dist.get_world_size())
    assert samples_per_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_per_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0


    for index in pbar:
        batch_images = []        

        for _ in range(4):
            z = torch.randn(
                (n, model.in_channels, latent_h, latent_w), 
                device=device, dtype=dtype
            )
            
            
            captions = all_captions[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]
            cap_features, cap_mask = encode_prompt(
                tokenizer, text_encoder, device, dtype, captions, 
                model_config.use_last_hidden_state, max_seq_length=model_config.max_seq_length
            )
                    
            cur_max_seq_len = cap_mask.sum(dim=-1).max()
            y = cap_features[:, : cur_max_seq_len]

            y_null = null_cap_feat[:, : cur_max_seq_len]
            y_null = y_null.expand(y.shape[0], cur_max_seq_len, null_cap_feat.shape[-1])
            

            # Sample images:
            with torch.no_grad():
                samples = scheduler.sample(
                    model, y, y_null, z, 
                    T_max=args.T_max, 
                    T_min=args.T_min,
                    num_steps=args.num_steps, 
                    cfg_scale=args.cfg_scale, 
                    cfg_low=args.guidance_low, 
                    cfg_high=args.guidance_high,
                    stochasticity_ratio=args.stochasticity_ratio,
                    sample_type=args.sample_type,
                )[-1]
                samples = samples.to(torch.float32)
            samples = decode_func(samples)
            batch_images.append(samples)
        paths = all_paths[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]
        metadatas = all_metadatas[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]
        for path in paths:
            os.makedirs(path, exist_ok=True)
            sample_path = os.path.join(path, 'samples')
            os.makedirs(sample_path, exist_ok=True)
        for i in range(len(paths)):
            with open(os.path.join(paths[i], "metadata.jsonl"), "w") as fp:
                json.dump(metadatas[i], fp)
        batch_images = torch.stack(batch_images, dim=1) 
        B, _, C, H, W = batch_images.shape
        batch_images = torch.clamp(127.5 * batch_images + 128.0, 0, 255).permute(0, 1, 3, 4, 2).to(torch.uint8).contiguous()
        # Save samples to disk as individual .png files
        for i, images in enumerate(batch_images.cpu().numpy()):
            for sample_count, image in enumerate(images):
                Image.fromarray(image).save(os.path.join(paths[i], 'samples', f"{sample_count:05}.png"))
            images = rearrange(images, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)
            Image.fromarray(images).save(os.path.join(paths[i], 'grid.png'))
        total += global_batch_size


    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--config", type=str, default=None, help="Optional config to a SiT checkpoint.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="")
    parser.add_argument("--data-type", type=str, default="geneval")
    parser.add_argument("--caption-dir", type=str, default="")


    # model
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--slice_vae", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--save-latent", action=argparse.BooleanOptionalAction, default=False)

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--T-max",  type=float, default=1.0)
    parser.add_argument("--T-min",  type=float, default=0.0)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--stochasticity-ratio", type=float, default=0.)
    parser.add_argument("--sample-type", type=str, default='transition')

    
    

    args = parser.parse_args()
    main(args)
