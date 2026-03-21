import math

import mlx.core as mx
import mlx.nn as nn

from .config import TadaConfig

__all__ = [
    "compute_llama3_inv_freq",
    "build_rope_cache",
    "apply_rope",
    "KVCache",
    "Attention",
    "MLP",
    "TransformerBlock",
    "LlamaModel",
]


def compute_llama3_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> mx.array:
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    old_ctx = original_max_position_embeddings
    low_wavelen = old_ctx / low_freq_factor
    high_wavelen = old_ctx / high_freq_factor
    new_freqs = []

    for i in range(inv_freq.shape[0]):
        freq = inv_freq[i].item()
        wavelen = 2 * math.pi / freq
        if wavelen < high_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_wavelen:
            new_freqs.append(freq / factor)
        else:
            smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

    return mx.array(new_freqs, dtype=mx.float32)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict,
) -> tuple[mx.array, mx.array]:
    inv_freq = compute_llama3_inv_freq(
        head_dim,
        rope_theta,
        rope_scaling["factor"],
        rope_scaling["low_freq_factor"],
        rope_scaling["high_freq_factor"],
        rope_scaling["original_max_position_embeddings"],
    )
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = mx.outer(positions, inv_freq)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    hdim = x.shape[-1]
    x0 = x[..., : hdim // 2]
    x1 = x[..., hdim // 2 :]
    cos = mx.expand_dims(mx.expand_dims(cos, 0), 0)
    sin = mx.expand_dims(mx.expand_dims(sin, 0), 0)
    return mx.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)


class KVCache:
    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset: int = 0

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is not None:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        else:
            self.keys = keys
            self.values = values

        self.offset = self.keys.shape[2]
        return self.keys, self.values

    @property
    def seq_len(self) -> int:
        return self.offset

    def clone(self) -> "KVCache":
        cloned = KVCache()
        cloned.keys = self.keys
        cloned.values = self.values
        cloned.offset = self.offset
        return cloned


class Attention(nn.Module):
    def __init__(self, config: TadaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        n_rep = self.num_heads // self.num_kv_heads

        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: TadaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TadaConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LlamaModel(nn.Module):
    def __init__(self, config: TadaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = inputs_embeds

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask, cache[i] if cache else None)

        return self.norm(h)
