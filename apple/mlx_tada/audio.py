import math

import numpy as np

try:
    import soundfile as sf_lib
except ImportError:
    sf_lib = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

__all__ = [
    "resample_audio",
    "load_audio",
    "transcribe_audio",
    "save_wav",
]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio

    assert resample_poly is not None, "Resampling requires scipy: pip install scipy"
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down, axis=-1).astype(np.float32)


def load_audio(audio_path: str, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    assert sf_lib is not None, "Audio loading requires soundfile: pip install soundfile"
    audio, sr = sf_lib.read(audio_path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = resample_audio(audio, sr, target_sr)
    return audio, target_sr


def transcribe_audio(audio_path: str) -> str:
    assert mlx_whisper is not None, (
        "audio_text not provided and mlx-whisper is not installed.\n"
        "Either provide audio_text or install mlx-whisper: pip install mlx-whisper"
    )
    result = mlx_whisper.transcribe(audio_path)
    return result["text"].strip()


def save_wav(wav: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Write a numpy audio array to a WAV file.

    Args:
        wav: Audio samples as numpy float32 array.
        path: Output file path.
        sample_rate: Sample rate in Hz. Default 24000.
    """
    assert sf_lib is not None, "Saving audio requires soundfile: pip install soundfile"
    audio_np = np.asarray(wav, dtype=np.float32)
    sf_lib.write(path, audio_np, sample_rate)
    print(f"Saved audio to {path} ({len(audio_np) / sample_rate:.2f}s)")
