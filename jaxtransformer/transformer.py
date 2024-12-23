import flax.linen as nn
import jax.numpy as jnp

global_dtype = jnp.bfloat16

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]
    
class Block(nn.Module):
    """ A standard transformer block. Has residual connection, self-attention, and a two-layer MLP. """
    hidden_size: int
    num_heads: int
    use_conditioning: bool
    use_causal_masking: bool
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        if self.use_conditioning:
            c = nn.silu(c) # Calculate adaLn modulation parameters.
            c = nn.Dense(6 * self.hidden_size)(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        else:
            shift_msa, shift_mlp = (jnp.zeros_like(x[:, 0]) for _ in range(2))
            scale_msa, gate_msa, scale_mlp, gate_mlp = (jnp.ones_like(x[:, 0]) for _ in range(4))
        
        # =========================
        # === Self-Attention Block. 
        # =========================
        x_norm = nn.LayerNorm(use_bias=False, use_scale=not self.use_conditioning, dtype=global_dtype)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        channels_per_head = self.hidden_size // self.num_heads
        k, q, v = [nn.Dense(self.hidden_size)(x_modulated) for _ in range(3)]
        k, q, v = [jnp.reshape(p, (k.shape[0], k.shape[1], self.num_heads, channels_per_head)) for p in [k, q, v]]
        q = q / jnp.sqrt(q.shape[3]) # 1/sqrt(d) scaling.
        w = jnp.einsum('bqhc,bkhc->bhqk', q, k) # [B, num_heads, Q, K]. Q,K = HW.
        w = w.astype(jnp.float32)
        if self.use_causal_masking:
            causal_mask = jnp.tri(N=w.shape[2], k=0) # [HW, HW].
            w = jnp.where(causal_mask[None, None, :, :], w, jnp.finfo(w.dtype).min)
            w = nn.softmax(w, axis=-1)
            w = jnp.where(causal_mask[None, None, :, :], w, 0)
        else:
            w = nn.softmax(w, axis=-1) # Softmax over key dimension = Total mass of 1 per query.
        y = jnp.einsum('bhqk,bkhc->bqhc', w, v) # [B, Q=HW, num_heads, channels_per_head]
        y = jnp.reshape(y, x.shape) # [B, Q=HW, C] (C = heads * channels_per_head)
        attn_x = nn.Dense(self.hidden_size)(y)
        attn_x = gate_msa[:, None] * attn_x
        x = x + attn_x

        # =========================
        # === MLP Block. 
        # =========================
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=not self.use_conditioning, dtype=global_dtype)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        y = nn.Dense(features=self.hidden_size * self.mlp_ratio)(x_modulated2)
        y = nn.gelu(y)
        mlp_x = nn.Dense(features=self.hidden_size)(y)
        mlp_x = gate_mlp[:, None] * mlp_x
        x = x + mlp_x
        return x

class TransformerBackbone(nn.Module):
    """Generic transformer backbone."""
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    use_conditioning: bool = False
    use_causal_masking: bool = False

    @nn.compact
    def __call__(self, x, c):
        assert len(x.shape) == 3 # Input tokens = (batch, seq_len, channels)
        assert len(c.shape) == 2 # Conditioning = (batch, channels)

        x = x.astype(global_dtype)
        c = c.astype(global_dtype)
        for _ in range(self.depth):
            x = Block(self.hidden_size, self.num_heads, self.use_conditioning, self.use_causal_masking, self.mlp_ratio)(x, c)
        return x