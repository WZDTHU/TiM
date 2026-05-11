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
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import functools
import argparse
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import StableDiffusion3Pipeline, FluxPipeline, SanaPipeline, SanaSprintPipeline, DiffusionPipeline

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
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    if 'sd3.5' in args.model_type:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    elif 'flux' in args.model_type:
        pipe = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    elif 'sana-sprint' in args.model_type:
        pipe = SanaSprintPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    elif 'sana' in args.model_type:
        pipe = SanaPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    elif 'hyper-sdxl' in args.model_type or 'sdxl-pcm' in args.model_type:
        pipe = DiffusionPipeline.from_pretrained('checkpoints/stable-diffusion-xl-base-1.0', torch_dtype=dtype)
        pipe.load_lora_weights(args.ckpt)
        pipe.fuse_lora()
    elif 'sdxl-turbo' in args.model_type or 'sdxl-lcm' in args.model_type:
        pipe = DiffusionPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    elif 'rectified-sdxl' in args.model_type:
        pipe = DiffusionPipeline.from_pretrained('checkpoints/stable-diffusion-xl-base-1.0', torch_dtype=dtype)
        weight = torch.load(args.ckpt, map_location="cpu")
        pipe.unet.load_state_dict(weight)
    
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
        
    
    # Create folder to save samples:
    folder_name = f"{args.data_type}-{args.model_type}-{args.height}x{args.width}-" \
                  f"Step-{args.num_steps}-cfg-{args.cfg_scale}" 
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
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_per_gpu = int(total_samples // dist.get_world_size())
    assert samples_per_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_per_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar


    for index in pbar:
        batch_images = []
        for _ in range(4):
            captions = all_captions[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]     
            with torch.no_grad():
                images = pipe(
                    prompt=captions,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.cfg_scale,
                ).images
                batch_images.append(np.asarray(images))
        batch_images = np.stack(batch_images, axis=1)

        metadatas = all_metadatas[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]
        paths = all_paths[samples_per_gpu*rank+n*index: samples_per_gpu*rank+n*(index+1)]
        for path in paths:
            os.makedirs(path, exist_ok=True)
            os.makedirs(os.path.join(path, 'samples'), exist_ok=True)
        # Save samples to disk as individual .png files
        for i, images in enumerate(batch_images):
            with open(os.path.join(paths[i], "metadata.jsonl"), "w") as fp:
                json.dump(metadatas[i], fp)
            for sample_count, image in enumerate(images):
                Image.fromarray(image).save(os.path.join(paths[i], 'samples', f"{sample_count:05}.png"))
            images = rearrange(images, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)
            Image.fromarray(images).save(os.path.join(paths[i], 'grid.png'))



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
    parser.add_argument("--model-type", type=str, default=None, help="Optional config to a SiT checkpoint.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="workdir/c2i/samples")
    parser.add_argument("--data-type", type=str, default="geneval")
    parser.add_argument("--caption-dir", type=str, default="projects/evaluate/geneval/prompts/evaluation_metadata.jsonl")


    # model
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    
    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=50)
    

    args = parser.parse_args()
    main(args)