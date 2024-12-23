import jax
import jax.numpy as jnp
import flax.linen as nn

def rms_norm(x, axis=None, keepdims=False):
    return jnp.sqrt(jnp.mean(x**2, axis=axis, keepdims=keepdims) + 1e-6)

def rms_normalize(x, axis=-1):
    norm = rms_norm(x, axis=axis, keepdims=True) + 1e-4
    return x / norm

def xavier_uniform_pytorchlike():
    def init(key, shape, dtype):
        from jax._src import core
        from jax._src import dtypes
        dtype = dtypes.canonicalize_dtype(dtype)
        if len(shape) == 2: # Dense, [in, out]
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # Conv, [k, k, in, out]. Assumes patch-embed style conv.
            fan_in = shape[0] * shape[1] * shape[2]
            fan_out = shape[3]
        else:
            raise ValueError(f"Invalid shape {shape}")

        variance = 2 / (fan_in + fan_out)
        scale = jnp.sqrt(3 * variance)
        param = jax.random.uniform(key, shape, dtype, -1) * scale

        return param
    return init

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]