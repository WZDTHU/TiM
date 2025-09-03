import torch
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.autoencoder_dc import Encoder, Decoder, AutoencoderDC


class MyEncoder(Encoder):
    def __init__(
        self, 
        in_channels, 
        latent_channels, 
        attention_head_dim = 32, 
        block_type = "ResBlock", 
        block_out_channels = ..., 
        layers_per_block = ..., 
        qkv_multiscales = ..., 
        downsample_block_type = "pixel_unshuffle", 
        out_shortcut = True
    ):
        super().__init__(
            in_channels, latent_channels, attention_head_dim, block_type, block_out_channels, 
            layers_per_block, qkv_multiscales, downsample_block_type, out_shortcut
        )

    def forward(self, hidden_states: torch.Tensor, use_checkpoint=False) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            if use_checkpoint:
                hidden_states = checkpoint(self.ckpt_wrapper(down_block), hidden_states)
            else:
                hidden_states = down_block(hidden_states)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    

    

class MyDecoder(Decoder):
    def __init__(
        self, 
        in_channels, 
        latent_channels, 
        attention_head_dim = 32, 
        block_type = "ResBlock", 
        block_out_channels = ..., 
        layers_per_block = ..., 
        qkv_multiscales = ..., 
        norm_type = "rms_norm", 
        act_fn = "silu", 
        upsample_block_type = "pixel_shuffle", 
        in_shortcut = True
    ):
        super().__init__(
            in_channels, latent_channels, attention_head_dim, block_type, block_out_channels, 
            layers_per_block, qkv_multiscales, norm_type, act_fn, upsample_block_type, in_shortcut
        )

    def forward(self, hidden_states: torch.Tensor, use_checkpoint=False) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(
                self.in_shortcut_repeats, dim=1, output_size=hidden_states.shape[1] * self.in_shortcut_repeats
            )
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in reversed(self.up_blocks):
            if use_checkpoint:
                hidden_states = checkpoint(self.ckpt_wrapper(up_block), hidden_states)
            else:
                hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    

class MyAutoencoderDC(AutoencoderDC):
    def __init__(
        self, 
        in_channels = 3, 
        latent_channels = 32, 
        attention_head_dim = 32, 
        encoder_block_types = "ResBlock", 
        decoder_block_types = "ResBlock", 
        encoder_block_out_channels = ..., 
        decoder_block_out_channels = ..., 
        encoder_layers_per_block = ..., 
        decoder_layers_per_block = ..., 
        encoder_qkv_multiscales = ..., 
        decoder_qkv_multiscales = ..., 
        upsample_block_type = "pixel_shuffle", 
        downsample_block_type = "pixel_unshuffle", 
        decoder_norm_types = "rms_norm", 
        decoder_act_fns = "silu", 
        scaling_factor = 1,
        bn_momentum = 0.1,
    ):
        super().__init__(
            in_channels, latent_channels, attention_head_dim, encoder_block_types, 
            decoder_block_types, encoder_block_out_channels, decoder_block_out_channels, 
            encoder_layers_per_block, decoder_layers_per_block, encoder_qkv_multiscales, 
            decoder_qkv_multiscales, upsample_block_type, downsample_block_type, 
            decoder_norm_types, decoder_act_fns, scaling_factor
        )

        self.encoder = MyEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
        )
        self.decoder = MyDecoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
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
        print(self.config.scaling_factor, self.bn.running_var.flatten())

    @property
    def mean(self):
        mean = self.bn.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return mean
    
    @property
    def std(self):
        std = self.bn.running_var.sqrt().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return std

    def forward(self, x: torch.Tensor, use_checkpoint=False) -> torch.Tensor:
        z = self.encoder(x, use_checkpoint)
        latent = self.bn(z)
        recon = self.decoder(z, use_checkpoint)
        posterior = None
        return posterior, latent, recon
