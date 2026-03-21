import argparse
import os

from .audio import save_wav
from .config import InferenceOptions, Reference
from .model import TadaForCausalLM, setup_logging

__all__ = [
    "main",
]


def main():
    parser = argparse.ArgumentParser(description="TADA inference on Apple Silicon via MLX")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--audio", type=str)
    parser.add_argument("--audio-text", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--acoustic-cfg", type=float, default=1.6)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--noise-temp", type=float, default=0.9)
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    args = parser.parse_args()

    if os.environ.get("DEBUG") == "1":
        setup_logging()

    if args.reference is None and args.audio is None:
        parser.error("Provide either --audio (+ optional --audio-text), or --reference (.npz)")

    model = TadaForCausalLM.from_weights(args.weights, quantize=args.quantize)

    if args.reference:
        reference = Reference.load(args.reference)
    else:
        reference = model.load_reference(args.audio, args.audio_text)

    opts = InferenceOptions(
        text_temperature=args.temperature,
        acoustic_cfg_scale=args.acoustic_cfg,
        num_flow_matching_steps=args.flow_steps,
        noise_temperature=args.noise_temp,
    )

    output = model.generate(args.text, reference, inference_options=opts)

    save_wav(output.audio, args.output)


if __name__ == "__main__":
    main()
