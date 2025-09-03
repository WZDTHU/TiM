import torch
from .dc_ae import MyAutoencoderDC as AutoencoderDC
from .sd_vae import MyAutoencoderKL as AutoencoderKL

# dc-ae
def get_dc_ae(vae_dir, dtype, device):
    dc_ae = AutoencoderDC.from_pretrained(vae_dir).to(dtype=dtype, device=device)
    dc_ae.eval()
    dc_ae.requires_grad_(False)
    return dc_ae


def dc_ae_encode(dc_ae, images):
    with torch.no_grad():
        z = dc_ae.encode(images).latent 
        latents = (z - dc_ae.mean) / dc_ae.std
    return latents

def dc_ae_decode(dc_ae, latents, slice_vae=False):
    with torch.no_grad():
        z =  latents * dc_ae.std + dc_ae.mean
        if slice_vae and z.size(0) > 1:
            decoded_slices = [dc_ae._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = dc_ae._decode(z)
        images = decoded    # decoded images
    return images

# sd-vae
def get_sd_vae(vae_dir, dtype, device):
    sd_vae = AutoencoderKL.from_pretrained(vae_dir).to(dtype=dtype, device=device)
    sd_vae.eval()
    sd_vae.requires_grad_(False)
    return sd_vae

def sd_vae_encode(sd_vae, images):
    with torch.no_grad():
        posterior = sd_vae.encode(images)
        z = posterior.latent_dist.sample()
        latents = (z - sd_vae.mean) / sd_vae.std
    return latents

def sd_vae_decode(sd_vae, latents, slice_vae=False):
    with torch.no_grad():
        z = latents * sd_vae.std + sd_vae.mean
        if slice_vae and z.shape[0] > 1:
            decoded_slices = [sd_vae._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = sd_vae._decode(z).sample
    return decoded



