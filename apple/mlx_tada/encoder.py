import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "create_segment_attention_mask",
    "Snake1d",
    "Conv1d",
    "ConvTranspose1d",
    "ResidualUnit",
    "EncoderBlock",
    "WavEncoder",
    "LocalSelfAttention",
    "LocalAttentionEncoderLayer",
    "LocalAttentionEncoder",
    "EncoderOutput",
    "Encoder",
]

HIDDEN_DIM = 1024
EMBED_DIM = 512
STRIDES = (6, 5, 4, 4)
NUM_ATTN_LAYERS = 6
NUM_ATTN_HEADS = 8
ATTN_DIM_FEEDFORWARD = 4096
ATTN_DROPOUT = 0.1
SAMPLE_STD = 0.5
ACOUSTIC_MEAN = 0.0
ACOUSTIC_STD = 1.5


def create_segment_attention_mask(text_token_mask: mx.array) -> mx.array:
    block_ids = mx.cumsum(text_token_mask, axis=1)
    block_i = mx.expand_dims(block_ids, 2)
    block_j = mx.expand_dims(block_ids, 1)
    same_block = block_i == block_j
    is_marked_i = mx.expand_dims(text_token_mask, 2).astype(mx.bool_)
    is_marked_j = mx.expand_dims(text_token_mask, 1).astype(mx.bool_)
    same_valid = same_block & (~is_marked_j | (is_marked_i & same_block))
    prev_block = block_j == (block_i - 1)
    prev_valid = prev_block & ~is_marked_j
    can_attend = same_valid | (is_marked_i & prev_valid)
    return ~can_attend


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x: mx.array) -> mx.array:
        return x + (1.0 / self.alpha) * mx.power(mx.sin(self.alpha * x), 2)


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        scale = math.sqrt(1.0 / (in_channels // groups * kernel_size))
        self.weight = mx.random.normal((out_channels, kernel_size, in_channels // groups)) * scale
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.normal((out_channels, kernel_size, in_channels)) * scale
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)

        if self.bias is not None:
            y = y + self.bias

        return y


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        diff = x.shape[1] - y.shape[1]

        if diff > 0:
            half = diff // 2
            x = x[:, half : half + y.shape[1], :]

        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.res1 = ResidualUnit(dim // 2, dilation=1)
        self.res2 = ResidualUnit(dim // 2, dilation=3)
        self.res3 = ResidualUnit(dim // 2, dilation=9)
        self.snake = Snake1d(dim // 2)
        self.conv = Conv1d(
            dim // 2,
            dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.snake(x)
        x = self.conv(x)
        return x


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64
        self.initial_conv = Conv1d(1, d_model, kernel_size=7, padding=3)
        self.blocks = []

        for stride in STRIDES:
            d_model *= 2
            self.blocks.append(EncoderBlock(d_model, stride=stride))

        self.final_snake = Snake1d(d_model)
        self.final_conv = Conv1d(d_model, HIDDEN_DIM, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.initial_conv(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.final_snake(x)
        x = self.final_conv(x)
        return x


class LocalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, self.head_dim, 2).astype(mx.float32) / self.head_dim))
        positions = mx.arange(max_seq_len).astype(mx.float32)
        freqs = mx.outer(positions, inv_freq)
        self._rope_cos = mx.cos(freqs)
        self._rope_sin = mx.sin(freqs)

    def apply_rope(self, x: mx.array, seq_len: int) -> mx.array:
        B, H, S, D = x.shape
        cos = mx.expand_dims(mx.expand_dims(self._rope_cos[:seq_len], 0), 0)
        sin = mx.expand_dims(mx.expand_dims(self._rope_sin[:seq_len], 0), 0)
        x_pairs = x.reshape(B, H, S, D // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        r0 = x0 * cos - x1 * sin
        r1 = x0 * sin + x1 * cos
        return mx.stack([r0, r1], axis=-1).reshape(B, H, S, D)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.apply_rope(q, S)
        k = self.apply_rope(k, S)
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            if mask.ndim == 2:
                attn = attn + mx.expand_dims(mx.expand_dims(mx.where(mask, mx.array(-1e9), mx.array(0.0)), 0), 0)

            elif mask.ndim == 3:
                attn = attn + mx.expand_dims(mx.where(mask, mx.array(-1e9), mx.array(0.0)), 1)

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)
        return self.layer_norm(x + out)


class LocalAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 4096):
        super().__init__()
        self.self_attn = LocalSelfAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.self_attn(x, mask)
        ffn_out = self.linear2(nn.gelu(self.linear1(x)))
        return self.norm(x + ffn_out)


class LocalAttentionEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 4, num_heads: int = 8, d_ff: int = 4096):
        super().__init__()
        self.layers = [LocalAttentionEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)

        return self.final_norm(x)


@dataclass
class EncoderOutput:
    audio_len: mx.array
    text: list[str]
    token_positions: mx.array
    token_values: mx.array
    sample_rate: int = 24000
    text_tokens: mx.array | None = None
    text_tokens_len: mx.array | None = None
    token_masks: mx.array | None = None


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav_encoder = WavEncoder()
        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=HIDDEN_DIM,
            num_layers=NUM_ATTN_LAYERS,
            num_heads=NUM_ATTN_HEADS,
            d_ff=ATTN_DIM_FEEDFORWARD,
        )
        self.hidden_linear = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.pos_emb = nn.Embedding(2, HIDDEN_DIM)

    def get_encoder_outputs(self, audio: mx.array, token_masks: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.expand_dims(audio, -1)
        x = mx.pad(x, [(0, 0), (0, 960), (0, 0)])
        enc_out = self.wav_encoder(x)
        seq_len = enc_out.shape[1]
        pad_len = seq_len - token_masks.shape[1]

        if pad_len > 0:
            padded_masks = mx.pad(token_masks, [(0, 0), (0, pad_len)])
        else:
            padded_masks = token_masks[:, :seq_len]

        enc_out = enc_out + self.pos_emb(padded_masks.astype(mx.int32))
        attn_mask = create_segment_attention_mask(padded_masks)
        enc_out = self.local_attention_encoder(enc_out, mask=attn_mask)
        enc_out = self.hidden_linear(enc_out)
        return enc_out, padded_masks

    def encode(
        self,
        audio: mx.array,
        token_positions: mx.array,
        token_masks: mx.array,
        audio_length: mx.array | None = None,
        text: list[str] | None = None,
        text_tokens: mx.array | None = None,
        text_tokens_len: mx.array | None = None,
        sample: bool = True,
    ) -> EncoderOutput:
        if audio_length is None:
            audio_length = mx.array([audio.shape[1]])

        enc_out, padded_masks = self.get_encoder_outputs(audio, token_masks)
        encoded_expanded = mx.where(
            mx.expand_dims(padded_masks, -1) == 0,
            mx.zeros_like(enc_out),
            enc_out,
        )

        if SAMPLE_STD > 0.0 and sample:
            noise = mx.random.normal(encoded_expanded.shape) * SAMPLE_STD
            encoded_expanded = mx.where(
                mx.expand_dims(padded_masks, -1) == 0,
                encoded_expanded,
                encoded_expanded + noise,
            )

        positions_clamped = mx.clip(token_positions - 1, 0, encoded_expanded.shape[1] - 1)
        B, T = positions_clamped.shape
        batch_idx = mx.repeat(mx.arange(B).reshape(-1, 1), T, axis=1)
        token_values = encoded_expanded[batch_idx, positions_clamped]
        token_values = (token_values - ACOUSTIC_MEAN) / ACOUSTIC_STD
        return EncoderOutput(
            audio_len=audio_length,
            text=text or [""],
            text_tokens=text_tokens,
            text_tokens_len=text_tokens_len,
            token_positions=token_positions,
            token_values=token_values,
            token_masks=padded_masks,
        )
