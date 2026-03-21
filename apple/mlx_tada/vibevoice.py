import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "TimestepEmbedder",
    "FeedForwardNetwork",
    "RMSNorm",
    "HeadLayer",
    "FinalLayer",
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceDiffusionHead",
]


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=False)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=False)

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half).astype(mx.float32) / half)
        args = mx.expand_dims(t.astype(mx.float32), 1) * mx.expand_dims(freqs, 0)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)

        return embedding.astype(t.dtype)

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(nn.silu(self.mlp_0(t_freq)))


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x_fp = x.astype(mx.float32)
        norm = x_fp * mx.rsqrt(mx.mean(x_fp * x_fp, axis=-1, keepdims=True) + self.eps)
        norm = norm.astype(dtype)

        if self.elementwise_affine:
            norm = norm * self.weight

        return norm


class HeadLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, cond_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation_linear = nn.Linear(cond_dim, 3 * embed_dim, bias=False)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.adaLN_modulation_linear(nn.silu(c))
        shift, scale, gate = mx.split(mod, 3, axis=-1)
        return x + gate * self.ffn(modulate(self.norm(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, cond_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation_linear = nn.Linear(cond_size, 2 * hidden_size, bias=False)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.adaLN_modulation_linear(nn.silu(c))
        shift, scale = mx.split(mod, 2, axis=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


@dataclass
class VibeVoiceDiffusionHeadConfig:
    hidden_size: int = 768
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-5
    latent_size: int = 64


class VibeVoiceDiffusionHead(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionHeadConfig):
        super().__init__()
        self.config = config
        cond_dim = config.hidden_size
        latent_size = config.latent_size

        self.noisy_images_proj = nn.Linear(latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(cond_dim)

        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)
        self.layers = [
            HeadLayer(
                embed_dim=config.hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=cond_dim,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.head_layers)
        ]
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            output_size=latent_size,
            cond_size=cond_dim,
            norm_eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        noisy_images: mx.array,
        timesteps: mx.array,
        condition: mx.array,
    ) -> mx.array:
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        conditioning = self.cond_proj(condition) + t

        for layer in self.layers:
            x = layer(x, conditioning)

        return self.final_layer(x, conditioning)

    @staticmethod
    def scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
        if schedule == "constant" or base_scale == 1.0:
            return base_scale

        if schedule == "linear":
            return 1.0 + (base_scale - 1.0) * (1.0 - t)

        if schedule == "cosine":
            return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))

        return base_scale

    @staticmethod
    def build_time_schedule(num_steps: int, schedule: str) -> mx.array:
        if schedule == "cosine":
            linear_steps = mx.linspace(0, 1, num_steps + 1)
            return 0.5 * (1 - mx.cos(math.pi * linear_steps))

        if schedule == "logsnr":
            log_snr = mx.linspace(5.0, -5.0, num_steps + 1)
            t_span = mx.sigmoid(-log_snr / 2)
            t_span = t_span.at[0].add(-t_span[0])
            t_span = t_span.at[-1].add(1.0 - t_span[-1])
            return t_span

        return mx.linspace(0, 1, num_steps + 1)

    def compute_velocity(
        self,
        speech: mx.array,
        t: mx.array,
        cond: mx.array,
        neg_cond: mx.array,
        acoustic_dim: int,
        acoustic_cfg: float,
        duration_cfg: float,
        bottleneck_fn=None,
    ) -> mx.array:
        apply_bn = bottleneck_fn or (lambda x: x)

        if acoustic_cfg != 1.0:
            speech_comb = mx.concatenate([speech, speech], axis=0)
            t_comb = mx.tile(t, (speech.shape[0] * 2,))
            cond_pos = mx.squeeze(cond, axis=1) if cond.ndim == 3 else cond
            cond_neg = mx.squeeze(neg_cond, axis=1) if neg_cond.ndim == 3 else neg_cond
            cond_comb = mx.concatenate([cond_pos, cond_neg], axis=0)
            vel_comb = self(speech_comb, t_comb, condition=apply_bn(cond_comb))
            vel_pos, vel_neg = mx.split(vel_comb, 2, axis=0)
            vel_acoustic = vel_neg[..., :acoustic_dim] + acoustic_cfg * (
                vel_pos[..., :acoustic_dim] - vel_neg[..., :acoustic_dim]
            )
            vel_time = vel_neg[..., acoustic_dim:] + duration_cfg * (
                vel_pos[..., acoustic_dim:] - vel_neg[..., acoustic_dim:]
            )
            return mx.concatenate([vel_acoustic, vel_time], axis=-1)

        cond_sq = mx.squeeze(cond, axis=1) if cond.ndim == 3 else cond
        return self(speech, mx.tile(t, (speech.shape[0],)), condition=apply_bn(cond_sq))

    def solve(
        self,
        noise: mx.array,
        cond: mx.array,
        neg_cond: mx.array,
        acoustic_dim: int,
        num_steps: int,
        acoustic_cfg_scale: float,
        duration_cfg_scale: float,
        cfg_schedule: str = "constant",
        time_schedule: str = "uniform",
        bottleneck_fn=None,
    ) -> mx.array:
        orig_dtype = noise.dtype
        speech = noise.astype(mx.float32)
        cond = cond.astype(mx.float32)
        neg_cond = neg_cond.astype(mx.float32)
        t_span_mx = self.build_time_schedule(num_steps, time_schedule)
        mx.eval(t_span_mx)
        t_span = [t_span_mx[i].item() for i in range(t_span_mx.shape[0])]

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_span[i - 1]
            t_val = t_span[i - 1]
            t_mx = mx.array(t_val, dtype=mx.float32)
            a_cfg = self.scheduled_cfg(acoustic_cfg_scale, t_val, cfg_schedule)
            d_cfg = self.scheduled_cfg(duration_cfg_scale, t_val, cfg_schedule)
            velocity = self.compute_velocity(speech, t_mx, cond, neg_cond, acoustic_dim, a_cfg, d_cfg, bottleneck_fn)
            speech = speech + dt * velocity

        return speech.astype(orig_dtype)
