# MLX-TADA

TADA speech synthesis on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Setup

```bash
cd apple
uv venv
uv pip install -e ".[convert]"
```

For auto-transcription of reference audio (optional):
```bash
uv pip install mlx-whisper
```

## Convert Weights

Requires a [Hugging Face](https://huggingface.co/) account with access to `meta-llama/Llama-3.2-1B` (gated model). Login first:
```bash
huggingface-cli login
```

Then convert:
```bash
# 3B model
uv run python -m mlx_tada.convert_3b ./weights/3b

# 1B model
uv run python -m mlx_tada.convert_1b ./weights/1b
```

## Generate Speech

### CLI

```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/1b \
  --audio speaker.wav \
  --text "Hello world, today is a nice day." \
  --output output.wav
```

With 4-bit quantization (10x faster, 60% less memory):
```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/1b \
  --audio speaker.wav \
  --text "Hello world, today is a nice day." \
  --quantize 4 \
  --output output.wav
```

### Python

```python
from mlx_tada import TadaForCausalLM

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("speaker.wav")
out = model.generate("Hello world, today is a nice day.", ref)

# out.audio     - numpy float32 array (24kHz)
# out.duration  - audio duration in seconds
# out.rtf       - real-time factor
# out.num_tokens
```

Save and reuse references:
```python
from mlx_tada import Reference

ref = model.load_reference("speaker.wav")
ref.save("speaker.npz")

ref = Reference.load("speaker.npz")
out = model.generate("Reusing the same voice.", ref)
```

Save audio:
```python
from mlx_tada import save_wav

save_wav(out.audio, "output.wav")
```

## Debug Logging

```bash
DEBUG=1 uv run python -m mlx_tada.generate \
  --weights ./weights/1b \
  --audio speaker.wav \
  --text "Hello"
```

```python
from mlx_tada import setup_logging

setup_logging()
```

## Running Tests

```bash
MLX_WEIGHTS=./weights/1b uv run pytest tests/ -v
```
