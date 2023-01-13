import flax.linen as nn
from einops import rearrange
import jax.numpy as jnp
from jax.numpy import einsum
from typing import Callable

ATTN_MASK_VALUE = -1e10

# functions and decorators

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# PreNorm

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

# Residual

class Residual(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        return x + y


# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    dim: int
    scale_base: int = 512
    use_xpos: bool = True         

    @nn.compact
    def __call__(self, seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        scale = (jnp.arange(0, self.dim, 2) + 0.4 * self.dim) / (1.4 * self.dim)

        seq = jnp.arange(seq_len)
        freqs = einsum("i , j -> i j", seq, inv_freq)
        freq = jnp.concatenate((freqs, freqs), axis = -1)

        if not self.use_xpos:
            return freqs, jnp.ones(1)

        power = (seq - (seq_len // 2 )) / self.scale_base
        scale = scale ** rearrange(power, 'n -> n 1')
        return freq, scale 

def rotate_half(x):
    x1, x2 = x.split(2, axis = -1)
    return jnp.concatenate((-x2, x1), axis = -1)

def apply_rotary_pos_emb(pos, t, scale = 1.):
    return (t * jnp.cos(pos) * scale) + (rotate_half(t) * jnp.sin(pos) * scale)

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class SwiGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x, gate = x.split(2, axis = -1)
        return jnp.multiply(nn.swish(gate), x)

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

class ParallelTransformerBlock(nn.Module):
    dim: int
    dim_head: int = 64
    causal: bool = True
    heads: int = 8
    ff_mult: int = 4
    attn_dropout = 0.
    ff_dropout = 0.
    use_xpos: bool = True
    xpos_scale_base: int = 512

    @nn.compact
    def __call__(
        self, 
        x, 
        mask = None, 
        finetune_modules = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        attn_inner_dim = self.dim_head * self.heads
        ff_inner_dim = self.dim * self.ff_mult
        fused_dims = (attn_inner_dim, self.dim_head, self.dim_head, (ff_inner_dim * 2))

        scale = self.dim_head ** -0.5

        n = x.shape[1]

        split_indices = numpy.cumsum(fused_dims[:-1])

        # attention queries, keys, values, and feedforward inner
        fused_attn_ff_proj = nn.Dense(features = sum(fused_dims), use_bias=False)(x)

        q, k, v, ff = jnp.split(fused_attn_ff_proj, split_indices, axis = -1)

        # finetune loras

        lora_q = lora_k = lora_v = lora_o = None

        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x)
            k = k + lora_k(x)
            v = v + lora_v(x)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h = self.heads)

        q = q * scale

        # rotary embeddings
        positions, scale = RotaryEmbedding(self.dim_head, scale_base = self.xpos_scale_base, use_xpos = self.use_xpos and self.causal)(n)

        if exists(positions) and positions.shape[-2] >= n:
            positions = positions[:n]
            scale = scale[:n]
            
        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        # similarity
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = jnp.where(~mask, sim, ATTN_MASK_VALUE)

        # causal mask

        if exists(mask) and mask.shape[-1] >= n:
            mask = mask[:n, :n]

        mask = jnp.ones((n, n)).triu(1)

        sim = jnp.where(mask, sim, ATTN_MASK_VALUE)

        # attention
        attn = nn.softmax(sim, axis = -1)
        attn = nn.Dropout(rate = self.attn_dropout)(attn)

        # aggregate values
        attn_out = einsum("b h i j, b j d -> b h i d", attn, v)

        # attention out
        out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = nn.Dense(self.dim, use_bias=False)(out)

        # feedforward out
        ff_out = SwiGLU()(ff)
        ff_out = nn.Dropout(rate = self.ff_dropout)(ff_out)
        ff_out = nn.Dense(self.dim, use_bias=False)(ff_out)

        # merge heads

        if exists(lora_o):
            attn_out = attn_out + lora_o(out)

        merge_heads = attn_out + ff_out
        return merge_heads

# cross attention block

class CrossAttention(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8
    dropout = 0.

    @nn.compact
    def __call__(
        self,
        x,
        kv,
        context_mask = None,
    ):
        scale = self.dim_head ** -0.5

        # queries, keys, values

        q, k, v = nn.Dense(features = self.inner_dim, use_bias=False)(x), *kv

        # split out heads and scale queries

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * scale

        # similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        attn = nn.softmax(sim, dim = -1)
        attn = nn.Dropout(rate = self.dropout)(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine heads

        return nn.Dense(features = self.dim, use_bias=False)(out)


# model

class PaLM(nn.Module): 
    dim: int 
    num_tokens: int
    depth: int 
    dim_head: int = 64 
    heads: int = 8 
    ff_mult: int = 4

    @nn.compact
    def __call__(self, x):
        embed = nn.Embed(self.num_tokens, self.dim, embedding_init = nn.initializers.normal(stddev=0.02))
        x = embed(x)
        x = ParallelTransformer(dim=self.dim, depth=self.depth, heads=self.heads, dim_head=self.dim_head, ff_mult=self.ff_mult)(x)
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        out = embed.attend(x)
        return out    


if __name__ == "__main__":

    import jax
    import numpy

    key = jax.random.PRNGKey(0)

    seq = jax.random.randint(key, (1, 2048), 0, 20000)

    model = PaLM(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2)}

    params = model.init(init_rngs, seq)
    output = model.apply(params, seq)
    print(output.shape) # (1, 2048, 20000)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}") # 55073280