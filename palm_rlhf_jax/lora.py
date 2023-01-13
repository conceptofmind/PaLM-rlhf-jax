import flax.linen as nn
import jax.numpy as jnp

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# LoRA - https://arxiv.org/abs/2106.09685

class LoRA(nn.Module):
    dim: int
    dim_out: int
    r: int = 8
    alpha = None

    @nn.compact
    def __call__(self, x):
        
        alpha = default(self.alpha, self.r)
        scale = alpha / self.r

        A = self.param('A', nn.initializers.normal(stddev = 0.02), (self.dim, self.r))
        B = self.param('B', nn.initializers.zeros, (self.r, self.dim_out))

        return jnp.dot(x, jnp.dot(A, B)) * scale