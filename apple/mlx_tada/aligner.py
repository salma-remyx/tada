import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .encoder import Conv1d

__all__ = [
    "align_text_tokens",
    "GroupNorm",
    "Wav2Vec2FeatureExtractorLayer",
    "Wav2Vec2FeatureExtractor",
    "Wav2Vec2FeatureProjection",
    "Wav2Vec2PositionalConvEmbedding",
    "Wav2Vec2Attention",
    "Wav2Vec2FeedForward",
    "Wav2Vec2EncoderLayer",
    "Wav2Vec2Encoder",
    "Wav2Vec2ForCTC",
    "Aligner",
]

HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 16
INTERMEDIATE_SIZE = 4096
NUM_HIDDEN_LAYERS = 24
CONV_DIM = (512, 512, 512, 512, 512, 512, 512)
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
NUM_CONV_POS_EMBEDDINGS = 128
NUM_CONV_POS_EMBEDDING_GROUPS = 16
VOCAB_SIZE = 128256


def align_text_tokens(probs: np.ndarray, text_tokens: np.ndarray) -> list[int]:
    L, V = probs.shape
    T = len(text_tokens)
    F = np.full((L, T), -np.inf, dtype=np.float32)
    backpointer = np.zeros((L, T), dtype=np.int64)
    token_probs = probs[:, text_tokens]
    cum_max_val = token_probs[0, 0]
    cum_max_idx = 0
    F[0, 0] = cum_max_val
    backpointer[0, 0] = 0

    for i in range(1, L):
        if token_probs[i, 0] > cum_max_val:
            cum_max_val = token_probs[i, 0]
            cum_max_idx = i
        F[i, 0] = cum_max_val
        backpointer[i, 0] = cum_max_idx

    if T <= L:
        cumsum = 0.0

        for k in range(T):
            cumsum += token_probs[k, k]
            F[k, k] = cumsum
            backpointer[k, k] = k

    for i in range(1, L):
        max_j = min(i, T)

        if max_j <= 1:
            continue

        for j in range(1, max_j):
            skip_score = F[i - 1, j]
            use_score = F[i - 1, j - 1] + token_probs[i, j]
            if use_score >= skip_score:
                F[i, j] = use_score
                backpointer[i, j] = i
            else:
                F[i, j] = skip_score
                backpointer[i, j] = -1

    positions = [0] * T
    i, j = L - 1, T - 1
    pos_idx = T - 1

    while j >= 0:
        if j == 0:
            positions[pos_idx] = int(backpointer[i, j])
            break
        elif backpointer[i, j] == -1:
            i -= 1
        else:
            positions[pos_idx] = int(backpointer[i, j])
            pos_idx -= 1
            i -= 1
            j -= 1

    return positions


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        C = x.shape[-1]
        G = self.num_groups
        spatial = x.shape[1:-1]
        x = x.reshape(B, -1, G, C // G)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, G, -1)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        x = x.reshape(B, G, -1, C // G)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, *spatial, C)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class Wav2Vec2FeatureExtractorLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, is_first: bool = False):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.is_first = is_first
        if is_first:
            self.layer_norm = GroupNorm(out_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.is_first:
            x = self.layer_norm(x)

        x = nn.gelu(x)
        return x


class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []
        in_ch = 1

        for i in range(len(CONV_DIM)):
            out_ch = CONV_DIM[i]
            kernel_size = CONV_KERNEL[i]
            stride = CONV_STRIDE[i]
            conv_layers.append(Wav2Vec2FeatureExtractorLayer(in_ch, out_ch, kernel_size, stride, is_first=(i == 0)))
            in_ch = out_ch

        self.conv_layers = conv_layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.conv_layers:
            x = layer(x)
        return x


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(CONV_DIM[-1])
        self.projection = nn.Linear(CONV_DIM[-1], HIDDEN_SIZE)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv1d(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            kernel_size=NUM_CONV_POS_EMBEDDINGS,
            padding=NUM_CONV_POS_EMBEDDINGS // 2,
            groups=NUM_CONV_POS_EMBEDDING_GROUPS,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = nn.gelu(x)
        return x


class Wav2Vec2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_ATTENTION_HEADS
        self.head_dim = HIDDEN_SIZE // NUM_ATTENTION_HEADS
        self.q_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.k_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.v_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.out_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_dense = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.output_dense = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = nn.gelu(x)
        x = self.output_dense(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Wav2Vec2Attention()
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.feed_forward = Wav2Vec2FeedForward()
        self.final_layer_norm = nn.LayerNorm(HIDDEN_SIZE)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        residual = x
        x = self.attention(x, mask)
        x = residual + x
        x = self.layer_norm(x)
        residual = x
        x = self.feed_forward(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class Wav2Vec2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding()
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.layers = [Wav2Vec2EncoderLayer() for _ in range(NUM_HIDDEN_LAYERS)]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        pos = self.pos_conv_embed(x)
        x = x + pos[:, : x.shape[1], :]
        x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class Wav2Vec2ForCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.feature_projection = Wav2Vec2FeatureProjection()
        self.encoder = Wav2Vec2Encoder()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def __call__(self, audio: mx.array) -> mx.array:
        x = mx.expand_dims(audio, -1)
        x = self.feature_extractor(x)
        x = self.feature_projection(x)
        x = self.encoder(x)
        return self.lm_head(x)

    def get_output_lengths(self, input_lengths: mx.array) -> mx.array:
        lengths = input_lengths

        for kernel_size, stride in zip(CONV_KERNEL, CONV_STRIDE):
            lengths = (lengths - kernel_size) // stride + 1

        return lengths


class Aligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForCTC()

    def __call__(
        self,
        audio_16k: mx.array,
        text_tokens: np.ndarray,
        input_lengths: np.ndarray,
        eos_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits = self.wav2vec2(audio_16k)
        max_input_length = int(input_lengths.max())
        batch_size = text_tokens.shape[0]
        all_token_positions = []
        all_token_masks = []

        for batch_idx in range(batch_size):
            batch_text_tokens = text_tokens[batch_idx]
            filtered_tokens = batch_text_tokens[batch_text_tokens != eos_token_id]
            valid_columns = np.concatenate([[0], filtered_tokens])
            valid_columns_mx = mx.array(valid_columns.astype(np.int32))
            logits_sparse = logits[batch_idx][:, valid_columns_mx].astype(mx.float32)
            mx.eval(logits_sparse)
            logits_sparse_np = np.array(logits_sparse)
            col_to_sparse = {int(v): i for i, v in enumerate(valid_columns)}
            sparse_text_tokens = np.array([col_to_sparse[int(t)] for t in filtered_tokens], dtype=np.int64)
            positions = align_text_tokens(logits_sparse_np, sparse_text_tokens)
            pos_emb = np.zeros(max_input_length, dtype=np.int64)
            pos_emb[positions] = 1
            positions_1indexed = np.array(positions, dtype=np.int64) + 1
            all_token_positions.append(positions_1indexed)
            all_token_masks.append(pos_emb)

        max_tokens = max(len(pos) for pos in all_token_positions)
        padded_positions = np.zeros((batch_size, max_tokens), dtype=np.int64)

        for batch_idx, pos in enumerate(all_token_positions):
            padded_positions[batch_idx, : len(pos)] = pos

        all_token_masks = np.stack(all_token_masks, axis=0)
        return padded_positions, all_token_masks
