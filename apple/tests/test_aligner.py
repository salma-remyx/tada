import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_tada.aligner import Aligner, align_text_tokens
from mlx_tada.audio import load_audio, resample_audio
from mlx_tada.model import load_weights

SAMPLE_DIR = Path(__file__).parent.parent.parent / "tada" / "samples"
WEIGHTS_DIR = Path(os.environ.get("MLX_WEIGHTS", "mlx_weights"))


@pytest.fixture(scope="module")
def aligner():
    aligner = Aligner()
    load_weights(aligner, WEIGHTS_DIR / "aligner" / "weights.safetensors")
    mx.eval(aligner.parameters())
    return aligner


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def test_align_text_tokens_basic():
    np.random.seed(42)
    probs = np.random.randn(10, 5).astype(np.float32)
    text_tokens = np.array([1, 3, 4], dtype=np.int64)
    positions = align_text_tokens(probs, text_tokens)
    assert len(positions) == 3
    assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))


def test_align_text_tokens_single_token():
    probs = np.zeros((5, 3), dtype=np.float32)
    probs[2, 1] = 10.0
    positions = align_text_tokens(probs, np.array([1], dtype=np.int64))
    assert positions == [2]


def test_aligner_ljspeech(aligner, tokenizer):
    audio_path = str(SAMPLE_DIR / "ljspeech.wav")
    text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."

    audio_24k, sr = load_audio(audio_path, target_sr=24000)
    audio_16k = resample_audio(audio_24k, 24000, 16000).reshape(1, -1)
    audio_len = len(audio_24k)

    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
    text_tokens_np = np.array([text_token_ids], dtype=np.int64)
    input_lengths = np.array([math.ceil(audio_len / sr * 50)], dtype=np.int64)

    token_positions, token_masks = aligner(
        mx.array(audio_16k),
        text_tokens_np,
        input_lengths,
        tokenizer.eos_token_id,
    )

    text_tokens_str = tokenizer.convert_ids_to_tokens(text_token_ids)

    assert list(zip(text_tokens_str, token_positions[0].tolist())) == [
        ("The", 2),
        ("Ġexamination", 31),
        ("Ġand", 47),
        ("Ġtestimony", 58),
        ("Ġof", 85),
        ("Ġthe", 92),
        ("Ġexperts", 114),
        (",", 149),
        ("Ġenabled", 156),
        ("Ġthe", 176),
        ("Ġcommission", 189),
        ("Ġto", 208),
        ("Ġconclude", 228),
        ("Ġthat", 266),
        ("Ġfive", 280),
        ("Ġshots", 298),
        ("Ġmay", 316),
        ("Ġhave", 323),
        ("Ġbeen", 332),
        ("Ġfired", 347),
        (".", 377),
    ]

    assert token_masks.shape[1] == input_lengths[0]
    assert token_masks.sum() == len(text_token_ids)


def test_aligner_logits_shape(aligner):
    audio_path = str(SAMPLE_DIR / "ljspeech.wav")
    audio_24k, _ = load_audio(audio_path, target_sr=24000)
    audio_16k = resample_audio(audio_24k, 24000, 16000).reshape(1, -1)

    logits = aligner.wav2vec2(mx.array(audio_16k))
    mx.eval(logits)

    assert logits.shape[0] == 1
    assert logits.shape[2] == 128256
    assert logits.shape[1] > 300


def test_aligner_output_masks_binary(aligner, tokenizer):
    audio_path = str(SAMPLE_DIR / "ljspeech.wav")
    text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."

    audio_24k, sr = load_audio(audio_path, target_sr=24000)
    audio_16k = resample_audio(audio_24k, 24000, 16000).reshape(1, -1)
    audio_len = len(audio_24k)

    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
    text_tokens_np = np.array([text_token_ids], dtype=np.int64)
    input_lengths = np.array([math.ceil(audio_len / sr * 50)], dtype=np.int64)

    _, token_masks = aligner(
        mx.array(audio_16k),
        text_tokens_np,
        input_lengths,
        tokenizer.eos_token_id,
    )

    unique_vals = set(token_masks.flatten().tolist())
    assert unique_vals <= {0, 1}
