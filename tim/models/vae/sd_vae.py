import torch
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.autoencoder_kl import Encoder, Decoder, AutoencoderKL
from typing import Optional

class MyEncoder(Encoder):
    def __init__(
        self, 
        in_channels = 3, 
        out_channels = 3, 
        down_block_types = ..., 
        block_out_channels = ..., 
        layers_per_block = 2, 
        norm_num_groups = 32, 
        act_fn = "silu", 
        double_z = True, 
        mid_block_add_attention=True
    ):
        super().__init__(
            in_channels, out_channels, down_block_types, block_out_channels, 
            layers_per_block, norm_num_groups, act_fn, double_z, mid_block_add_attention
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # down
            for down_block in self.down_blocks:
                sample = checkpoint(self.ckpt_wrapper(down_block), sample)
            # middle
            sample = checkpoint(self.ckpt_wrapper(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward


class MyDecoder(Decoder):
    def __init__(
        self, 
        in_channels = 3, 
        out_channels = 3, 
        up_block_types = ..., 
        block_out_channels = ..., 
        layers_per_block = 2, 
        norm_num_groups = 32, 
        act_fn = "silu", 
        norm_type = "group", 
        mid_block_add_attention=True
    ):
        super().__init__(
            in_channels, out_channels, up_block_types, block_out_channels, 
            layers_per_block, norm_num_groups, act_fn, norm_type, mid_block_add_attention
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # middle
            sample = checkpoint(self.ckpt_wrapper(self.mid_block), sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = checkpoint(self.ckpt_wrapper(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward


class MyAutoencoderKL(AutoencoderKL):
    def __init__(
        self, 
        in_channels = 3, 
        out_channels = 3, 
        down_block_types = ..., 
        up_block_types = ..., 
        block_out_channels = ..., 
        layers_per_block = 1, 
        act_fn = "silu", 
        latent_channels = 4, 
        norm_num_groups = 32, 
        sample_size = 32, 
        scaling_factor = 0.18215, 
        shift_factor = None, 
        latents_mean = None, 
        latents_std = None, 
        force_upcast = True, 
        use_quant_conv = True, 
        use_post_quant_conv = True, 
        mid_block_add_attention = True,
        bn_momentum = 0.1, 
    ):
        super().__init__(
            in_channels, out_channels, down_block_types, up_block_types, block_out_channels, 
            layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, 
            scaling_factor, shift_factor, latents_mean, latents_std, force_upcast, 
            use_quant_conv, use_post_quant_conv, mid_block_add_attention
        )
        self.encoder = MyEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = MyDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.bn = torch.nn.BatchNorm2d(
            latent_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True
        )
        self.bn.reset_running_stats()
        self.init_bn()
        

    def init_bn(self):
        # self.bn.running_mean = torch.zeros_like(self.bn.running_mean).to(torch.float64)
        # self.bn.running_var = torch.ones_like(self.bn.running_var).to(torch.float64) / self.config.scaling_factor ** 2
        self.bn.running_mean = torch.zeros_like(self.bn.running_mean)
        self.bn.running_var = torch.ones_like(self.bn.running_var) / self.config.scaling_factor ** 2

    @property
    def mean(self):
        mean = self.bn.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return mean
    
    @property
    def std(self):
        std = self.bn.running_var.sqrt().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return std
    

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        self.encoder.gradient_checkpointing = use_checkpoint
        self.decoder.gradient_checkpointing = use_checkpoint
        posterior = self.encode(x).latent_dist
        z = posterior.sample()
        latent = self.bn(z)
        recon = self.decode(z).sample
        return posterior, latent, recon


        
