# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.layers.mlp import SwiGLU, Mlp  
from timm.models.vision_transformer import PatchEmbed, Attention
from tim.models.utils.funcs import build_mlp, modulate, get_parameter_dtype
from tim.models.utils.rope import VisionRotaryEmbedding, rotate_half
from flash_attn import flash_attn_func


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cap_feat_dim, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(cap_feat_dim)
        self.mlp = SwiGLU(in_features=cap_feat_dim, hidden_features=hidden_size*4, out_features=hidden_size)
        

    def forward(self, cap_feats):
        '''
        cfg is also essential in text-to-image generation
        '''
        cap_feats = self.mlp(self.norm(cap_feats))
        return cap_feats



#################################################################################
#                                 Attention Block                               #
#################################################################################

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            distance_aware: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.distance_aware = distance_aware
        if distance_aware:
            self.qkv_d = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, freqs_cos, freqs_sin, attn_type='fused_attn', delta_t=None) -> torch.Tensor:
        B, N, C = x.shape
        if self.distance_aware:
            qkv = self.qkv(x) + self.qkv_d(delta_t)
        else:
            qkv = self.qkv(x)
        if attn_type == 'flash_attn':   # q, k, v: (B, N, n_head, d_head)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        else:                           # q, k, v: (B, n_head, N, d_head)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        ori_dtype = qkv.dtype
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        k = k * freqs_cos + rotate_half(k) * freqs_sin
        q, k = q.to(ori_dtype), k.to(ori_dtype)

        if attn_type == 'flash_attn':
            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.reshape(B, N, C)
        elif attn_type == 'fused_attn':
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x






#################################################################################
#                              Cross Attention Block                            #
#################################################################################

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, freqs_cos, freqs_sin, attn_type='fused_attn') -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = y.shape
        if attn_type == 'flash_attn':   # q, k, v: (B, N, n_head, d_head)
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
            kv = self.kv(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        else:                           # q, k, v: (B, n_head, N, d_head)
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        ori_dtype = q.dtype
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        q, k = q.to(ori_dtype), k.to(ori_dtype)

        if attn_type == 'flash_attn':
            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.reshape(B, N, C)
        elif attn_type == 'fused_attn':
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x






#################################################################################
#                                 Core TiM Model                                #
#################################################################################

class TiMBlock(nn.Module):
    """
    A TiM block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        distance_aware = block_kwargs.get('distance_aware', False)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"],
            distance_aware=distance_aware
        )
        self.norm2_i = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_t = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(
            in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=True
        )
        if block_kwargs.get('lora_hidden_size', None) != None:
            lora_hidden_size = block_kwargs['lora_hidden_size']
        else: 
            lora_hidden_size = (hidden_size//4)*3
        self.adaLN_modulation = SwiGLU(
            in_features=hidden_size, hidden_features=lora_hidden_size, out_features=9*hidden_size, bias=True
        )
        
        

    def forward(self, x, y, c, freqs_cos, freqs_sin, attn_type, delta_t=None):
        (
            shift_msa, scale_msa, gate_msa, 
            shift_msc, scale_msc, gate_msc, 
            shift_mlp, scale_mlp, gate_mlp 
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cos, freqs_sin, attn_type, delta_t)
        x = x + gate_msc * self.cross_attn(modulate(self.norm2_i(x), shift_msc, scale_msc), self.norm2_t(y), freqs_cos, freqs_sin, attn_type)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of TiM.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = SwiGLU(
            in_features=hidden_size, hidden_features=hidden_size//2, out_features=2*hidden_size, bias=True
        )
        

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class TiM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cap_feat_dim=2048,
        z_dim=768,
        projector_dim=2048,
        use_checkpoint: bool = False,
        new_condition: str = 't-r',
        use_new_embed: bool = False,
        **block_kwargs # qk_norm
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cap_feat_dim = cap_feat_dim
        self.encoder_depth = encoder_depth
        self.use_checkpoint = use_checkpoint
        self.new_condition = new_condition
        self.use_new_embed = use_new_embed

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        if use_new_embed:
            self.delta_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = CaptionEmbedder(cap_feat_dim, hidden_size)
        # Will use fixed sin-cos embedding:
        self.rope = VisionRotaryEmbedding(head_dim=hidden_size//num_heads)

        self.blocks = nn.ModuleList([
            TiMBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.projector = build_mlp(hidden_size, projector_dim, z_dim) 
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.mlp.fc1_g.weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp.fc1_x.weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp.fc2.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in TiM blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)
            

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation.fc2.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.fc2.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, H, W):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h, w = int(H/p), int(W/p)
        

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def get_rope(self, h, w, attn_type):
        grid_h = torch.arange(h)
        grid_w = torch.arange(w)
        grid = torch.meshgrid(grid_h, grid_w, indexing='xy') 
        grid = torch.stack(grid, dim=0).reshape(2, -1).unsqueeze(0)
        freqs_cos, freqs_sin = self.rope.get_cached_2d_rope_from_grid(grid)
        if attn_type == 'flash_attn':   # (1, N, 1, d_head)
            return freqs_cos.unsqueeze(2), freqs_sin.unsqueeze(2)
        else:                           # (1, 1, N, d_head)
            return freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            

    def forward(self, x, t, r, y, attn_type='flash_attn', return_zs=False, jvp=False):
        """
        Forward pass of TiM.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        B, C, H, W = x.shape
        x = self.x_embedder(x)                          # (N, N, D), where T = H * W / patch_size ** 2

        # timestep and class embedding
        t_embed = self.t_embedder(t).unsqueeze(1)                   # (B, 1, D)
        delta_embed = self.get_delta_embed(t, r).unsqueeze(1)       # (B, 1, D)
        y = self.y_embedder(y)                                      # (B, M, D)
        c = t_embed + delta_embed                                   # (B, 1, D)


        freqs_cos, freqs_sin = self.get_rope(
            int(H/self.patch_size), int(W/self.patch_size), attn_type
        )      

        for i, block in enumerate(self.blocks):
            if not self.use_checkpoint or jvp:
                x = block(x, y, c, freqs_cos, freqs_sin, attn_type, delta_embed)   # (B, N, D)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, y, c, freqs_cos, freqs_sin, attn_type, delta_embed
                )
            if (i + 1) == self.encoder_depth:
                h_proj = self.projector(x)
        x = self.final_layer(x, c)                      # (B, N, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, H, W)                    # (b, out_channels, H, W)

        if return_zs:
            return x, h_proj
        else:
            return x
        
    def get_delta_embed(self, t, r):
        if self.use_new_embed:
            delta_embedder = self.delta_embedder
        else:
            delta_embedder = self.t_embedder
        if self.new_condition == 't-r':
            delta_embed = delta_embedder(t-r)
        elif self.new_condition == 'r':
            delta_embed = delta_embedder(r)
        elif self.new_condition == 't,r':
            delta_embed = self.t_embedder(t) + delta_embedder(r)
        elif self.new_condition == 't,t-r':
            delta_embed = self.t_embedder(t) + delta_embedder(t-r)
        elif self.new_condition == 'r,t-r':
            delta_embed = self.t_embedder(r) + delta_embedder(t-r)
        elif self.new_condition == 't,r,t-r':
            delta_embed = self.t_embedder(t) + self.t_embedder(r) + delta_embedder(t-r)
        else:
            raise NotImplementedError
        return delta_embed
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    
   
  