import math

import mlx.core as mx
import mlx.nn as nn

from .encoder import (
    Conv1d,
    ConvTranspose1d,
    LocalAttentionEncoder,
    ResidualUnit,
    Snake1d,
)

__all__ = [
    "create_segment_attention_mask",
    "DecoderBlock",
    "DACDecoder",
    "Decoder",
]

EMBED_DIM = 512
HIDDEN_DIM = 1024
NUM_ATTN_LAYERS = 6
NUM_ATTN_HEADS = 8
ATTN_DIM_FEEDFORWARD = 4096
WAV_DECODER_CHANNELS = 1536
STRIDES = (4, 4, 5, 6)


def create_segment_attention_mask(text_token_mask: mx.array) -> mx.array:
    block_ids = mx.cumsum(text_token_mask, axis=1) - text_token_mask
    block_i = mx.expand_dims(block_ids, 2)
    block_j = mx.expand_dims(block_ids, 1)
    same_block = block_i == block_j
    prev_block = block_j == (block_i - 1)
    can_attend = same_block | prev_block
    return ~can_attend


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.snake = Snake1d(input_dim)
        self.conv_transpose = ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res1 = ResidualUnit(output_dim, dilation=1)
        self.res2 = ResidualUnit(output_dim, dilation=3)
        self.res3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake(x)
        x = self.conv_transpose(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: tuple[int, ...] = STRIDES,
        d_out: int = 1,
    ):
        super().__init__()
        self.initial_conv = Conv1d(input_channel, channels, kernel_size=7, padding=3)
        self.blocks = []

        for i, stride in enumerate(rates):
            in_dim = channels // 2**i
            out_dim = channels // 2 ** (i + 1)
            self.blocks.append(DecoderBlock(in_dim, out_dim, stride))

        final_dim = channels // 2 ** len(rates)
        self.final_snake = Snake1d(final_dim)
        self.final_conv = Conv1d(final_dim, d_out, kernel_size=7, padding=3)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.initial_conv(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.final_snake(x)
        x = self.final_conv(x)
        return mx.tanh(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_proj = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=HIDDEN_DIM,
            num_layers=NUM_ATTN_LAYERS,
            num_heads=NUM_ATTN_HEADS,
            d_ff=ATTN_DIM_FEEDFORWARD,
        )
        self.wav_decoder = DACDecoder(input_channel=HIDDEN_DIM, channels=WAV_DECODER_CHANNELS, rates=STRIDES)

    def __call__(self, encoded_expanded: mx.array, token_masks: mx.array) -> mx.array:
        x = self.decoder_proj(encoded_expanded)
        attn_mask = create_segment_attention_mask(token_masks)
        x = self.local_attention_decoder(x, mask=attn_mask)
        return self.wav_decoder(x)

    def decode_frames(self, encoded: mx.array, time_before: mx.array) -> mx.array:
        T = encoded.shape[0]
        if T == 0:
            return mx.zeros((0,))
        time_before = time_before[: T + 1]
        parts = []

        for pos in range(T):
            n_zeros = max(0, int(time_before[pos].item()) - 1)
            if n_zeros > 0:
                parts.append(mx.zeros((n_zeros, encoded.shape[-1])))
            parts.append(mx.expand_dims(encoded[pos], 0))

        n_trailing = int(time_before[-1].item())

        if n_trailing > 0:
            parts.append(mx.zeros((n_trailing, encoded.shape[-1])))

        expanded = mx.concatenate(parts, axis=0)
        expanded = mx.expand_dims(expanded, 0)
        token_masks = (mx.sqrt(mx.sum(expanded * expanded, axis=-1)) != 0).astype(mx.int32)
        wav = self(expanded, token_masks)
        mx.eval(wav)
        return wav.reshape(-1)
