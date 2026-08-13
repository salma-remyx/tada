"""Run the SpeechR speech-reasoning benchmark against a TADA model.

Wires :mod:`tada.eval.speech_reasoning` onto TADA's audio path, following
the same ``run_*_tada.py`` convention as the hellaswag / storycloze / sSC
evals: load ``TadaForCausalLM``, load the codec ``Encoder``, iterate a
SpeechR-format manifest, score each candidate answer by conditional loss,
and print per-category accuracy alongside the overall headline number.

SpeechR (arXiv:2508.02018) evaluates reasoning *over speech* -- temporal,
spatial, speaker, and content reasoning -- via multiple-choice questions
about an audio clip. This runner consumes a JSONL manifest of such items:

    {"audio": "path/to/clip.wav", "question": "Who spoke second?",
     "candidates": ["Alice", "Bob", "Carol"], "label": 1,
     "category": "speaker"}

The dataset itself and its official task splits are a downstream concern;
this script provides the scoring harness that runs them through TADA.
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tada.eval.speech_reasoning import SpeechRItem, aggregate_by_category, score_item
from tada.modules.encoder import Encoder
from tada.modules.tada import InferenceOptions, TadaForCausalLM


def load_manifest(path: str) -> list[SpeechRItem]:
    """Load a JSONL manifest of SpeechR items (one JSON object per line)."""
    items: list[SpeechRItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(
                SpeechRItem(
                    audio=record["audio"],
                    question=record["question"],
                    candidates=record["candidates"],
                    label=int(record["label"]),
                    category=record.get("category", "uncategorized"),
                    sample_rate=int(record.get("sample_rate", 24000)),
                )
            )
    return items


def evaluate_speechr(
    model_id: str = "HumeAI/tada-1b",
    manifest_path: str = "speechr.jsonl",
    text_only: bool = False,
    limit: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "text-only" if text_only else "audio"
    print(f"Using device: {device}")
    print(f"Mode: {mode}  Model: {model_id}  Manifest: {manifest_path}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = TadaForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    model.to(device).eval()

    encoder = None
    if not text_only:
        encoder = Encoder.from_pretrained("HumeAI/tada-codec").to(device).eval()

    def tokenize(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    items = load_manifest(manifest_path)
    if limit is not None:
        items = items[:limit]
    print(f"Loaded {len(items)} SpeechR items")

    results = []
    for item in tqdm(items, desc="Evaluating SpeechR"):
        results.append(
            score_item(
                item,
                model,
                tokenize,
                device,
                encoder=encoder,
                inference_options=InferenceOptions(acoustic_cfg_scale=1.0),
            )
        )

    stats = aggregate_by_category(results)
    print("\nSpeechR results by category:")
    for category, bucket in sorted(stats.items()):
        print(f"  {category:<14} {bucket.correct}/{bucket.total} = {bucket.accuracy * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SpeechR speech-reasoning benchmark against TADA.")
    parser.add_argument("--model_id", default="HumeAI/tada-1b")
    parser.add_argument("--manifest_path", default="speechr.jsonl")
    parser.add_argument("--text_only", action="store_true", help="Score text-only conditions (zeroed acoustics).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    args = parser.parse_args()

    evaluate_speechr(
        model_id=args.model_id,
        manifest_path=args.manifest_path,
        text_only=args.text_only,
        limit=args.limit,
    )
