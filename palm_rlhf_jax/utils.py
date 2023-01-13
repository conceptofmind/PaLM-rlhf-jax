import math
import flax.linen as nn
import jax
import jax.numpy as jnp

from einops import rearrange

def exists(val):
    return val is not None

# decorators

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# tensor helpers

def log(t, eps = 1e-20):
    return jnp.log(jax.lax.clamp(min = eps, x = t))

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if not exists(mask):
        return seq.mean(axis = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = jnp.where(~mask, seq, 0.)
    numer = masked_seq.sum(axis = dim, keepdims = keepdim)
    denom = mask.sum(axis = dim, keepdims = keepdim)

    masked_mean = numer / jax.lax.clamp(min = 1e-3, x = denom)
    masked_mean = jnp.where(denom == 0, masked_mean, 0.)
    return masked_mean

# sampling helpers

def gumbel_noise(t):
    noise = jnp.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# def top_p(logits, thres = 0.9):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cum_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

#     sorted_indices_to_remove = cum_probs > (1 - thres)
#     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
#     sorted_indices_to_remove[:, 0] = 0

#     sorted_logits[sorted_indices_to_remove] = float('-inf')
#     return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# def top_k(logits, thres = 0.9):
#     k = math.ceil((1 - thres) * logits.shape[-1])
#     val, ind = torch.topk(logits, k)
#     probs = torch.full_like(logits, float('-inf'))
#     probs.scatter_(1, ind, val)
#     return probs