import jax
import math
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from jaxtransformer.transformer import global_dtype, modulate

################################################################################
#                                 Input Modules                                #
################################################################################

class TimestepEmbed(nn.Module):
    """ Embeds scalar continuous time into vector representations."""
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), dtype=global_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x
    
    def timestep_embedding(self, t, max_period=10000):
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(global_dtype) * jnp.sqrt(2) # RMS norm = 1.
        return embedding


class TokenEmbed(nn.Module):
    """ Embed integer tokens into vector representations. """
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, labels):
        embedding_table = nn.Embed(self.num_classes, self.hidden_size, 
                                embedding_init=nn.initializers.normal(0.02), dtype=global_dtype)
        return embedding_table(labels)
    
class PatchEmbed(nn.Module):
    """ Embed 2D image into a sequence of tokens. """
    patch_size: int
    hidden_size: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.hidden_size, patch_tuple, patch_tuple, use_bias=self.bias, padding="VALID",
                     dtype=global_dtype)(x) # (B, P, P, hidden_size)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x
    
# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32))
    return jnp.expand_dims(emb,0)

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0) # (1, H*W, D)
    
################################################################################
#                                Output Modules                                #
################################################################################
    
class PatchOutput(nn.Module):
    """ Final layer for image-output models. """
    patch_size: int
    channels: int

    @nn.compact
    def __call__(self, x):
        batch_size, num_patches, _ = x.shape
        patch_side = int(num_patches ** 0.5)
        x = nn.Dense(self.patch_size * self.patch_size * self.channels, dtype=global_dtype)(x)
        x = jnp.reshape(x, (batch_size, patch_side, patch_side, self.patch_size, self.patch_size, self.channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=patch_side, W=patch_side)
        return x
    
class ClassifierOutput(nn.Module):
    """ Final layer for classification (e.g. image classification, language modelling)"""
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(use_bias=False, use_scale=False, dtype=global_dtype)(x)
        return nn.Dense(self.num_classes, global_dtype)(x)
