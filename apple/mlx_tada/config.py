from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "TadaConfig",
    "InferenceOptions",
    "Reference",
    "GenerationOutput",
]


@dataclass
class TadaConfig:
    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 131072
    acoustic_dim: int = 512
    num_time_classes: int = 256
    shift_acoustic: int = 5
    head_layers: int = 6
    head_ffn_ratio: float = 4.0
    bottleneck_dim: int | None = None
    acoustic_mean: float = 0.0
    acoustic_std: float = 1.5
    bos_token_id: int = 128000
    eos_token_id: list[int] = field(default_factory=lambda: [128001, 128008, 128009])
    pad_token_id: int = 128004
    start_header_id: int = 128006
    end_header_id: int = 128007
    eot_id: int = 128009

    @classmethod
    def from_dict(cls, d: dict) -> "TadaConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}

        if "eos_token_id" in filtered and isinstance(filtered["eos_token_id"], int):
            filtered["eos_token_id"] = [filtered["eos_token_id"]]

        return cls(**filtered)


@dataclass
class InferenceOptions:
    """Parameters controlling speech generation behavior.

    Key parameters:
        text_temperature: Controls randomness of text token sampling. Lower = more deterministic.
        acoustic_cfg_scale: Classifier-free guidance scale for acoustic features.
            Higher = better text adherence, lower = more natural prosody.
        num_flow_matching_steps: Number of Euler steps in the diffusion ODE solver.
            Fewer = faster but lower quality. 20 is default, 10 is a good balance, 5 is near real-time.
        noise_temperature: Scale of initial noise for flow matching. Lower = less variation.
    """

    text_do_sample: bool = True
    text_temperature: float = 0.6
    text_top_k: int = 0
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.1
    acoustic_cfg_scale: float = 1.6
    duration_cfg_scale: float = 1.0
    cfg_schedule: Literal["constant", "linear", "cosine"] = "cosine"
    noise_temperature: float = 0.9
    num_flow_matching_steps: int = 20
    time_schedule: Literal["uniform", "cosine", "logsnr"] = "logsnr"
    num_acoustic_candidates: int = 1
    scorer: Literal["spkr_verification", "likelihood", "duration_median"] = "likelihood"
    negative_condition_source: Literal["negative_step_output", "prompt", "zero"] = "negative_step_output"
    text_only_logit_scale: float = 0.0


@dataclass
class Reference:
    """Encoded reference audio for voice cloning.

    Created by TadaForCausalLM.load_reference(). Can be saved to disk
    with save() and loaded with Reference.load() for reuse.
    """

    token_values: np.ndarray
    token_positions: np.ndarray
    token_masks: np.ndarray | None
    text_tokens: np.ndarray
    text_tokens_len: np.ndarray
    audio_len: np.ndarray
    text: str
    sample_rate: int = 24000

    def save(self, path: str) -> None:
        """Save this reference to a .npz file for later reuse."""
        data = {
            "token_values": self.token_values,
            "token_positions": self.token_positions,
            "text_tokens": self.text_tokens,
            "text_tokens_len": self.text_tokens_len,
            "audio_len": self.audio_len,
            "text": np.array([self.text]),
            "sample_rate": np.array([self.sample_rate]),
        }

        if self.token_masks is not None:
            data["token_masks"] = self.token_masks

        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "Reference":
        """Load a previously saved reference from a .npz file."""
        data = np.load(path, allow_pickle=True)
        text = data["text"]

        if isinstance(text, np.ndarray):
            text = str(text.flat[0])

        sr = data.get("sample_rate", np.array([24000]))

        if isinstance(sr, np.ndarray):
            sr = int(sr.flat[0])

        return cls(
            token_values=data["token_values"],
            token_positions=data["token_positions"],
            token_masks=data.get("token_masks"),
            text_tokens=data["text_tokens"],
            text_tokens_len=data["text_tokens_len"],
            audio_len=data["audio_len"],
            text=text,
            sample_rate=sr,
        )


@dataclass
class GenerationOutput:
    """Output from TadaForCausalLM.generate().

    Attributes:
        audio: Raw waveform as numpy float32 array at 24kHz.
        num_tokens: Number of acoustic tokens generated.
        duration: Audio duration in seconds.
        rtf: Real-time factor (generation_time / audio_duration). Below 1.0 is faster than real-time.
    """

    audio: np.ndarray
    num_tokens: int
    duration: float
    rtf: float
